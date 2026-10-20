"""Voxel topology optimisation: 3D SIMP compliance minimisation with 8-node hexahedral elements (CPU, scipy only).

A Problem is a voxel design space plus supports and loads. optimize() keeps a target share of the initial mass where it stiffens the
part the most and reports each step of the pipeline through an optional callback, so a UI can show it live.
"""
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

try:  # multi-threaded MKL solver: several times faster than SuperLU on a multi-core CPU; optional
    from pypardiso import PyPardisoSolver
except Exception:  # any import problem just means: use SuperLU
    PyPardisoSolver = None

FREE, VOID, SOLID = -1, 0, 1  # design codes, the same as the SELTO dataset
CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
EMIN = 1e-6  # stiffness of void relative to solid


@dataclass
class Problem:
    name: str
    design: np.ndarray  # (nx, ny, nz) int8: FREE (optimised), VOID (no material), SOLID (kept as is)
    fixed: np.ndarray  # (nx, ny, nz, 3) bool: the nodes of these voxels cannot move along x / y / z
    force: np.ndarray  # (nx, ny, nz, 3) float: force in newtons on each loaded voxel, shared by its 8 nodes
    voxel_mm: float = 1.0
    young_mpa: float = 70_000.0
    density_g_cm3: float = 2.7
    keep: float = 0.3  # share of the initial mass (free + solid voxels) to keep
    radius: float = 1.5  # density filter radius in voxels: sets the smallest feature size


@dataclass
class Result:
    structure: np.ndarray  # (nx, ny, nz) bool, the final part
    density: np.ndarray  # (nx, ny, nz) float, filtered density at the end (0 outside the design space)
    history: list  # per iteration: dict(it, compliance, mass, change, seconds)
    snapshots: list  # (iteration, bool structure) taken along the way
    metrics: dict
    energy: np.ndarray = field(default=None)  # (nx, ny, nz) strain energy per voxel of the final part, relative to its mean


def hex8_stiffness(nu=0.3):
    """24 x 24 stiffness matrix of a unit cube with E = 1 (2 x 2 x 2 Gauss points)."""
    lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 0.5 / (1 + nu)
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[range(3), range(3)] += 2 * mu
    D[range(3, 6), range(3, 6)] = mu
    g, Ke = 1 / np.sqrt(3), np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for ze in (-g, g):
                c = CORNERS
                dN = 0.125 * 2 * np.stack([c[:, 0] * (1 + c[:, 1] * eta) * (1 + c[:, 2] * ze),
                                           c[:, 1] * (1 + c[:, 0] * xi) * (1 + c[:, 2] * ze),
                                           c[:, 2] * (1 + c[:, 0] * xi) * (1 + c[:, 1] * eta)], 1)  # d/dx of the 8 shape functions
                B = np.zeros((6, 24))
                for n, (dx, dy, dz) in enumerate(dN):
                    B[0, 3 * n], B[1, 3 * n + 1], B[2, 3 * n + 2] = dx, dy, dz
                    B[3, 3 * n], B[3, 3 * n + 1] = dy, dx
                    B[4, 3 * n + 1], B[4, 3 * n + 2] = dz, dy
                    B[5, 3 * n], B[5, 3 * n + 2] = dz, dx
                Ke += B.T @ D @ B * 0.125
    return Ke


def nested_dissection(coords, leaf=48):
    """Node order that keeps the fill-in of a grid-like mesh small: each half first, the plane that separates them last.

    Nodes of a hexahedral mesh only touch nodes one step away, so removing a plane of nodes really splits the mesh in two.
    """
    order = []

    def rec(idx):
        if len(idx) <= leaf:
            return order.append(idx)
        c = coords[idx]
        axis = np.ptp(c, axis=0).argmax()
        mid = np.sort(c[:, axis])[len(idx) // 2]
        rec(idx[c[:, axis] < mid])
        rec(idx[c[:, axis] > mid])
        order.append(idx[c[:, axis] == mid])

    rec(np.arange(len(coords)))
    return np.concatenate(order)


class Model:
    """Finite-element model of a Problem: active elements, node numbering, loads, supports and the fixed sparse pattern."""

    def __init__(self, prob, nu=0.3):
        self.shape = nx, ny, nz = prob.design.shape
        self.eid = np.flatnonzero((prob.design != VOID).ravel())
        n = self.n = len(self.eid)
        ijk = np.stack(np.unravel_index(self.eid, self.shape), 1)
        self.centroid = ijk + 0.5
        sy, sx = nz + 1, (ny + 1) * (nz + 1)
        nodes = ijk[:, 0:1] * sx + ijk[:, 1:2] * sy + ijk[:, 2:3] + np.array([0, sx, sx + sy, sy, 1, sx + 1, sx + sy + 1, sy + 1])
        used = np.zeros((nx + 1) * sx, bool)
        used[nodes] = True
        enodes = (np.cumsum(used) - 1).astype(np.int32)[nodes]  # compact node ids, (n, 8)
        self.ndof = 3 * int(used.sum())
        self.edof = (3 * enodes[:, :, None] + np.arange(3, dtype=np.int32)).reshape(n, 24)
        self.Ke = hex8_stiffness(nu)
        fixed, f = np.zeros(self.ndof, bool), np.zeros(self.ndof)
        flags, forces = prob.fixed.reshape(-1, 3)[self.eid], prob.force.reshape(-1, 3)[self.eid]
        for d in range(3):
            fixed[3 * enodes[flags[:, d]].ravel() + d] = True
            m = forces[:, d] != 0
            f += np.bincount((3 * enodes[m] + d).ravel(), weights=np.repeat(forces[m, d] / 8, 8), minlength=self.ndof)
        self.free, self.f = ~fixed, f
        coords = np.stack(np.unravel_index(np.flatnonzero(used), (nx + 1, ny + 1, nz + 1)), 1)
        order = (3 * nested_dissection(coords)[:, None] + np.arange(3)).ravel()
        order = self.order = order[self.free[order]]  # free dofs, node by node in nested-dissection order
        rank = np.empty(self.ndof, np.int32)
        rank[order] = np.arange(len(order))
        fmap = np.where(self.free, rank, -1).astype(np.int32)[self.edof]  # free-dof ids per element dof
        nf = self.nfree = int(self.free.sum())
        valid = (fmap[:, :, None] >= 0) & (fmap[:, None, :] >= 0)
        key = (fmap[:, :, None].astype(np.int64) * nf + fmap[:, None, :])[valid]
        uniq, self.inv = np.unique(key, return_inverse=True)
        rows = uniq // nf
        self.valid, self.indices = valid.ravel(), (uniq % nf).astype(np.int32)
        self.indptr = np.searchsorted(rows, np.arange(nf + 1)).astype(np.int32)
        self.upper = self.indices >= rows  # PARDISO only needs the upper triangle of a symmetric matrix
        self.upper_indices = self.indices[self.upper]
        self.upper_indptr = np.searchsorted(rows[self.upper], np.arange(nf + 1)).astype(np.int32)
        self.pardiso, self.analysed = None, False

    def solve(self, E):
        """Displacements (unit Young's modulus scale) for the element stiffness factors E, shape (n,)."""
        data = np.bincount(self.inv, weights=np.multiply.outer(E, self.Ke.ravel()).ravel()[self.valid], minlength=len(self.indices))
        u, b = np.zeros(self.ndof), self.f[self.order]
        if PyPardisoSolver and self.pardiso is not False:
            try:  # symmetric positive definite solver; the analysis is done once, later iterations only factorise again
                if self.pardiso is None:
                    self.pardiso = PyPardisoSolver(mtype=2)
                A = sparse.csr_matrix((data[self.upper], self.upper_indices, self.upper_indptr), shape=(self.nfree, self.nfree))
                self.pardiso.set_phase(22 if self.analysed else 12)
                self.pardiso._call_pardiso(A, np.zeros((self.nfree, 1)))
                self.analysed = True
                self.pardiso.set_phase(33)
                u[self.order] = self.pardiso._call_pardiso(A, b.reshape(-1, 1)).ravel()
                return u
            except Exception:  # PARDISO unavailable or refused the matrix: use SuperLU from here on
                self.close()
                self.pardiso = False
        # the dofs are numbered in nested-dissection order already, so SuperLU keeps that order and prefers the diagonal as pivot
        K = sparse.csc_matrix((data, self.indices, self.indptr), shape=(self.nfree, self.nfree))  # symmetric: CSR arrays read as CSC are fine
        u[self.order] = splu(K, permc_spec="NATURAL", diag_pivot_thresh=0.01, options=dict(SymmetricMode=True)).solve(b)
        return u

    def close(self):
        """Release the memory PARDISO keeps for its factorisation."""
        if self.pardiso:
            try:
                self.pardiso.free_memory(everything=True)
            except Exception:  # nothing left to free
                pass
        self.analysed = False

    def energy(self, u):
        """Strain energy of each element for unit stiffness: u_e' Ke u_e."""
        ue = u[self.edof]
        return np.einsum("ij,ij->i", ue @ self.Ke, ue)

    def grid(self, values, fill=0.0):
        g = np.full(int(np.prod(self.shape)), fill, dtype=np.result_type(values, float) if values.dtype != bool else bool)
        g[self.eid] = values
        return g.reshape(self.shape)

    def filter(self, radius):
        """Density filter H (n x n) and its row sums: neighbours within radius, weighted linearly by distance."""
        pairs = cKDTree(self.centroid).query_pairs(radius, output_type="ndarray")
        w = radius - np.linalg.norm(self.centroid[pairs[:, 0]] - self.centroid[pairs[:, 1]], axis=1)
        i, j, k = np.r_[pairs[:, 0], pairs[:, 1], np.arange(self.n)], np.r_[pairs[:, 1], pairs[:, 0], np.arange(self.n)], np.r_[w, w, np.full(self.n, radius)]
        H = sparse.csr_matrix((k, (i, j)), shape=(self.n, self.n))
        return H, np.asarray(H.sum(1)).ravel()


def validate(prob):
    """Raise a readable error if the problem cannot be solved."""
    act = prob.design != VOID
    if not (prob.fixed & act[..., None]).any():
        raise ValueError("no supports: the part would float away")
    if not (np.abs(prob.force).sum(-1) > 0)[act].any():
        raise ValueError("no loads to carry")
    lab, n = ndimage.label(act, np.ones((3, 3, 3)))
    for i in range(1, n + 1):
        if not (prob.fixed.any(-1) & (lab == i)).any():
            raise ValueError("part of the design space is not connected to any support")
    solid = int((prob.design == SOLID).sum())
    free = int((prob.design == FREE).sum())
    if not 0 < prob.keep * act.sum() - solid < free:
        raise ValueError(f"mass to keep must lie between {solid / act.sum():.0%} (fixed parts only) and 100%")


def pick(density, free, n_free_keep):
    """Mask that keeps the n_free_keep densest free elements (exact mass, whatever the threshold would give)."""
    keep = np.zeros(len(density), bool)
    idx = np.flatnonzero(free)
    keep[idx[np.argsort(-density[idx], kind="stable")[:n_free_keep]]] = True
    return keep


def optimize(prob, iterations=50, snapshots=10, callback=None, nu=0.3, penal=3.0):
    """Minimise compliance for prob.keep of the mass. callback(stage, **info) is told about each pipeline step:
    model, iteration (many), extract, verify. penal: final SIMP penalty, higher gives crisper (less grey) designs."""
    validate(prob)
    t0 = time.perf_counter()
    model = Model(prob, nu)
    try:
        return _run(prob, model, t0, iterations, snapshots, callback or (lambda *a, **k: None), penal)
    finally:
        model.close()


def _run(prob, model, t0, iterations, snapshots, emit, penal):
    design = prob.design.ravel()[model.eid]
    solid, free = design == SOLID, design == FREE
    n, n_keep = model.n, int(round(prob.keep * model.n))
    emit("model", elements=n, dofs=model.ndof, supported=int(prob.fixed.any(-1)[prob.design != VOID].sum()),
         loaded=int((np.abs(prob.force).sum(-1) > 0).sum()), solid=int(solid.sum()), seconds=time.perf_counter() - t0)

    H, Hs = model.filter(prob.radius)
    dv = H.T @ (1 / Hs)
    x = np.where(solid, 1.0, (n_keep - solid.sum()) / free.sum())
    xp = H @ x / Hs
    history, snaps, every = [], [], max(1, iterations // snapshots)
    for it in range(1, iterations + 1):
        ts = time.perf_counter()
        p = min(penal, 2.0 + (penal - 2.0) * it / (0.4 * iterations))  # penalty 2 -> penal over the first 40% of the iterations
        u = model.solve(EMIN + (1 - EMIN) * xp**p)
        ce = model.energy(u)
        compliance = float(model.f @ u)
        dc = H.T @ (-p * (1 - EMIN) * xp ** (p - 1) * ce / Hs)
        lo, hi = 1e-12, 1e12  # optimality-criteria update: bisection on the Lagrange multiplier so the mass hits its target
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            xn = np.clip(x * np.sqrt(np.maximum(-dc, 1e-30) / (dv * mid)), np.maximum(x - 0.2, 0), np.minimum(x + 0.2, 1))
            xn[solid] = 1
            lo, hi = (mid, hi) if (H @ xn / Hs).sum() > n_keep else (lo, mid)
            if hi / lo < 1 + 1e-6:
                break
        change, x = float(np.abs(xn - x).max()), xn
        xp = H @ x / Hs
        history.append(dict(it=it, compliance=compliance, mass=float(xp.sum() / n), change=change, seconds=time.perf_counter() - ts))
        emit("iteration", total=iterations, **history[-1])
        if it % every == 0 or it == iterations:
            snaps.append((it, model.grid(solid | pick(xp, free, n_keep - int(solid.sum())))))
        if it > iterations // 2 and change < 0.01:
            break

    # extract the part: exact mass, then drop pieces that carry nothing
    keep = solid | pick(xp, free, n_keep - int(solid.sum()))
    grid = model.grid(keep)
    lab, _ = ndimage.label(grid, ndimage.generate_binary_structure(3, 1))
    anchored = np.unique(lab[prob.fixed.any(-1) & grid])
    clean = np.isin(lab, anchored[anchored > 0])
    emit("extract", removed=int(grid.sum() - clean.sum()), kept=int(clean.sum()))

    # verify with the same finite-element model: the final part against the full design space
    metrics, energy = assess(prob, model, clean)
    metrics.update(iterations=len(history), seconds=time.perf_counter() - t0, solid_share=float(solid.sum() / n))
    emit("verify", **metrics)
    return Result(clean, model.grid(xp * clean.ravel()[model.eid]), history, snaps, metrics, energy)


def assess(prob, model, structure):
    """Mass and stiffness of a boolean structure compared with the full design space, with the same finite-element model.
    Returns (metrics, strain energy per voxel relative to its mean)."""
    n, cell = model.n, structure.ravel()[model.eid]
    u1, u0 = model.solve(EMIN + (1 - EMIN) * cell), model.solve(np.ones(n))
    c1, c0 = float(model.f @ u1), float(model.f @ u0)
    vox_cm3 = (prob.voxel_mm / 10) ** 3
    scale = 1 / (prob.young_mpa * prob.voxel_mm)  # unit-E, unit-cube result -> newton-millimetres
    m0, m1 = n * vox_cm3 * prob.density_g_cm3, int(cell.sum()) * vox_cm3 * prob.density_g_cm3
    e = model.energy(u1) * cell
    nodes = np.unique(model.edof[cell][:, 0::3] // 3)  # nodes of the part
    peak = lambda u: float(np.linalg.norm(u.reshape(-1, 3)[nodes], axis=1).max())
    metrics = dict(elements=n, mass_initial_g=m0, mass_final_g=m1, mass_saved=1 - m1 / m0, compliance_initial=c0 * scale,
                   compliance_final=c1 * scale, stiffness_retained=c0 / c1, stiffness_per_mass_gain=(c0 / c1) / (m1 / m0),
                   max_disp_ratio=peak(u1) / peak(u0))
    return metrics, model.grid(e / max(e[cell].mean(), 1e-30))


def evaluate(prob, structure, nu=0.3):
    """Finite-element check of any boolean structure, for example an ML prediction: the same metrics as optimize() reports."""
    model = Model(prob, nu)
    try:
        return assess(prob, model, structure & (prob.design != VOID))[0]
    finally:
        model.close()
