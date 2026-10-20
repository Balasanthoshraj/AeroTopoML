"""Mechanical check: a small voxel finite-element solver, and reference vs predicted structure of the SELTO test samples.

  python src/fem.py        solves the reference and the predicted structure of the 200 test samples
                           -> outputs/fem_per_sample.csv and outputs/fem_summary.json

Model: linear elasticity, one trilinear 8-node hexahedron per voxel (nodes on the voxel corners, 40 x 40 x 5 nodes), full 2 x 2 x 2
Gauss integration, sparse direct solve. Void voxels keep E_VOID of the solid stiffness, so floating nodes do not make the system singular.
Data conventions (prepare.py arrays):
  * dirichlet: a flagged voxel fixes all 8 of its corner nodes in the flagged directions.
  * force_val is a BODY FORCE DENSITY in N/m^3 (SELTO Zenodo record: "All forces are body forces given in [N/m^3]"; paper section 3),
    constant over the voxel. A voxel therefore carries F * h^3 newtons, an eighth on each corner node. This gives total loads of
    1500 to 2000 N per sample, round numbers; read as newtons per voxel they would be about 1e11 N.
Compliance C = f.u (joule) for the same load f on both structures, so the stiffness ratio predicted / reference is C_ref / C_pred.
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.stats import spearmanr

import aerotopoml as A

E_VOID = 1e-9  # stiffness of a void voxel relative to a solid one
CORNERS = np.array(list(itertools.product((0, 1), repeat=3)))  # corner offsets of a voxel = its local node order
HIGH_DICE, TOL = 0.90, 0.10  # "high overlap" and "stiffness differs" thresholds for the summary


# ---------------------------------------------------------------- solver
def element_stiffness(nu):
    """24 x 24 stiffness of a unit-edge trilinear hexahedron with E = 1 (a cube of edge h and modulus E: times E * h)."""
    lam, mu = nu / (1 + nu) / (1 - 2 * nu), 0.5 / (1 + nu)
    D = np.diag([lam + 2 * mu] * 3 + [mu] * 3)
    D[:3, :3] += lam * (1 - np.eye(3))
    s, K = 2 * CORNERS - 1, np.zeros((24, 24))  # corners in natural coordinates (-1 / +1)
    for xi in itertools.product((-1, 1), repeat=3):  # Gauss points +-1/sqrt(3), weight 1
        t = 1 + s * np.array(xi) / np.sqrt(3)
        dN = 0.25 * s * t[:, [1, 2, 0]] * t[:, [2, 0, 1]]  # dN_a/dx_c for x = (xi + 1) / 2
        B = np.zeros((6, 8, 3))
        for c in range(3):
            B[c, :, c] = dN[:, c]
            B[3 + c, :, (c + 1) % 3], B[3 + c, :, (c + 2) % 3] = dN[:, (c + 2) % 3], dN[:, (c + 1) % 3]  # gamma_yz, gamma_zx, gamma_xy
        B = B.reshape(6, 24)
        K += B.T @ D @ B / 8  # det J = 1/8
    return K


def to_nodes(vox):
    """Mark the 8 corner nodes of every flagged voxel: bool array (nx, ny, nz, ...) -> (nx + 1, ny + 1, nz + 1, ...)."""
    nx, ny, nz = vox.shape[:3]
    out = np.zeros((nx + 1, ny + 1, nz + 1, *vox.shape[3:]), bool)
    for i, j, k in CORNERS:
        out[i:nx + i, j:ny + j, k:nz + k] |= vox
    return out


def assemble(solid, nu):
    """Global stiffness for E = 1 and h = 1 (CSR), voxels in C order = x-major, z fastest, like the data."""
    nx, ny, nz = solid.shape
    nid = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nx + 1, ny + 1, nz + 1)
    nodes = np.stack([nid[i:nx + i, j:ny + j, k:nz + k].ravel() for i, j, k in CORNERS], 1)  # (voxels, 8)
    dof = (3 * nodes[:, :, None] + np.arange(3)).reshape(-1, 24)
    vals = np.where(solid.ravel(), 1.0, E_VOID)[:, None, None] * element_stiffness(nu)
    rows, cols = np.broadcast_to(dof[:, :, None], vals.shape), np.broadcast_to(dof[:, None, :], vals.shape)
    return sparse.coo_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(3 * nid.size,) * 2).tocsr()


def solve(solid, fixed, f, E, nu, h):
    """Nodal displacements u (m), shape (nx + 1, ny + 1, nz + 1, 3), of the voxel grid `solid` (bool) under the nodal loads f (N, same
    shape) with the degrees of freedom flagged in `fixed` held at zero. Raises if the residual is not small."""
    free = ~fixed.ravel()
    K = assemble(solid, nu)[free][:, free].tocsc()
    b = f.ravel()[free]
    # K is symmetric positive definite, so no pivoting is needed (1.5x faster)
    uf = splu(K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)).solve(b)
    res = np.linalg.norm(K @ uf - b) / np.linalg.norm(b)
    if not res < 1e-8:
        raise RuntimeError(f"FE solve failed: relative residual {res:.1e} (structure without supports?)")
    u = np.zeros(f.size)
    u[free] = uf / (E * h)  # K_physical = E * h * K_unit
    return u.reshape(f.shape)


# ---------------------------------------------------------------- SELTO samples
def nodal_loads(shape, idx, val, h):
    """Body force densities val (n, 3) in N/m^3 in the voxels idx (n, 3) -> nodal loads in N: F * h^3 per voxel, an eighth per corner."""
    f = np.zeros((*(s + 1 for s in shape), 3))
    for corner in CORNERS:
        np.add.at(f, tuple((idx + corner).T), val * h ** 3 / 8)
    return f


def sample_problem(d, i):
    """Fixed degrees of freedom and nodal loads of sample i of a split loaded with aerotopoml.load."""
    m = d["force_idx"][:, 0] == i
    return to_nodes(d["dirichlet"].astype(bool)), nodal_loads(A.SHAPE, d["force_idx"][m, 1:].astype(int), d["force_val"][m], d["info"][3])


def mechanics(solid, fixed, f, info):
    """Compliance (J) and largest displacement magnitude (m, over the nodes that touch material) of one structure."""
    u = solve(solid, fixed, f, info[0], info[1], info[3])
    return float((f * u).sum()), float(np.linalg.norm(u, axis=-1)[to_nodes(solid)].max())


# ---------------------------------------------------------------- reference vs prediction
def spread(x):
    """Distribution of a per-sample ratio."""
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(mean=x.mean(), min=x.min(), p5=q[0], p25=q[1], median=q[2], p75=q[3], p95=q[4], max=x.max())


def summarise(df, info):
    """Summary of the per-sample table: ratio distributions, shares within a tolerance, rank correlations, and the samples whose overlap
    is high although the stiffness differs."""
    off = (df.stiffness_ratio - 1).abs()
    high = df.dice_free >= HIGH_DICE
    odd = df[high & (off > TOL)].sort_values("stiffness_ratio")
    bins = pd.cut(df.dice_free, [0, HIGH_DICE, 0.95, 1.01], right=False, labels=[f"<{HIGH_DICE}", f"{HIGH_DICE}-0.95", ">=0.95"])
    by_overlap = {str(k): dict(samples=len(g), median_stiffness_ratio=g.stiffness_ratio.median(), share_within_10pct=float((off[g.index] <= TOL).mean()),
                               min_stiffness_ratio=g.stiffness_ratio.min(), max_stiffness_ratio=g.stiffness_ratio.max())
                  for k, g in df.groupby(bins, observed=True)}
    return dict(
        samples=len(df), voxel_elements=int(np.prod(A.SHAPE)), void_stiffness=E_VOID,
        material=dict(zip(("E_Pa", "nu", "yield_Pa", "h_m"), map(float, info))),
        force_unit="body force density N/m^3 (SELTO), voxel load = F * h^3, an eighth per corner node",
        total_load_N=dict(values=sorted(df.load_N.round(1).unique()), min=df.load_N.min(), max=df.load_N.max()),
        reference=dict(compliance_J=spread(df.compliance_ref_J), maxdisp_mm=spread(df.maxdisp_ref_mm)),
        stiffness_ratio_pred_over_ref=spread(df.stiffness_ratio), maxdisp_ratio_pred_over_ref=spread(df.maxdisp_ratio),
        volume_ratio_pred_over_ref=spread(df.volume_ratio), stiffness_per_volume_ratio=spread(df.stiffness_per_volume_ratio),
        share_within={f"{t:.0%}": float((off <= t).mean()) for t in (0.05, 0.10, 0.20)},
        share_maxdisp_within={f"{t:.0%}": float(((df.maxdisp_ratio - 1).abs() <= t).mean()) for t in (0.10, 0.20)},
        share_prediction_stiffer=float((df.stiffness_ratio > 1).mean()),
        spearman=dict(dice_free_vs_abs_log_stiffness_ratio=spearmanr(df.dice_free, np.log(df.stiffness_ratio).abs())[0],
                      log_volume_ratio_vs_log_stiffness_ratio=spearmanr(np.log(df.volume_ratio), np.log(df.stiffness_ratio))[0]),
        by_dice_free=by_overlap,
        high_overlap_but_different=dict(
            rule=f"dice_free >= {HIGH_DICE} and |stiffness ratio - 1| > {TOL}", high_overlap_samples=int(high.sum()), count=len(odd),
            stiffer=int((odd.stiffness_ratio > 1).sum()), beyond_20pct=int((off[high] > 0.20).sum()),
            samples=[dict(test_sample=int(i), dice_free=r.dice_free, stiffness_ratio=r.stiffness_ratio, volume_ratio=r.volume_ratio)
                     for i, r in odd.iterrows()]))


def main():
    t0 = time.perf_counter()
    d = A.load("test")
    pipe, cfg = A.load_pipeline()
    pred = A.structures(pipe.density(d["x"]), d, **cfg["post"])
    rows = []
    for i in range(len(pred)):
        fixed, f = sample_problem(d, i)
        (c_ref, u_ref), (c_pred, u_pred) = (mechanics(s, fixed, f, d["info"]) for s in (d["density"][i], pred[i]))
        rows.append(dict(load_N=np.linalg.norm(f.sum((0, 1, 2))), ref_voxels=d["density"][i].sum(), pred_voxels=pred[i].sum(),
                         compliance_ref_J=c_ref, compliance_pred_J=c_pred, maxdisp_ref_mm=1e3 * u_ref, maxdisp_pred_mm=1e3 * u_pred))
        if i % 20 == 0:
            print(f"sample {i}: stiffness ratio {c_ref / c_pred:.3f}, {time.perf_counter() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df.index.name = "test_sample"
    df["dice_free"] = A.overlap(pred, d["density"], d["design"] == A.FREE)[0]
    df["stiffness_ratio"] = df.compliance_ref_J / df.compliance_pred_J  # same load on both: stiffer prediction = ratio above 1
    df["maxdisp_ratio"] = df.maxdisp_pred_mm / df.maxdisp_ref_mm
    df["volume_ratio"] = df.pred_voxels / df.ref_voxels
    df["stiffness_per_volume_ratio"] = df.stiffness_ratio / df.volume_ratio  # material efficiency of the prediction relative to the reference
    df.to_csv(A.OUT / "fem_per_sample.csv", float_format="%.6g")
    summary = summarise(df, d["info"]) | dict(runtime_s=time.perf_counter() - t0)
    (A.OUT / "fem_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    shown = ("stiffness_ratio_pred_over_ref", "maxdisp_ratio_pred_over_ref", "share_within", "spearman")
    print(json.dumps({k: summary[k] for k in shown}, indent=1, default=float))


if __name__ == "__main__":
    main()
