"""Self-checks for the voxel FE solver: element matrix, exact uniform tension, void padding, beam convergence, unit conversion, monotonicity.

  python tests/test_fem.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aerotopoml as A  # noqa: E402
import fem  # noqa: E402

E, NU, H = 70e9, 0.3, 0.01  # Pa, -, m


def tension_block(sigma, nx=6, ny=3, nz=2):
    """Solid block pulled by the uniform stress sigma on its x = L face; u_x = 0 on x = 0 plus the minimum rigid-body constraints."""
    fixed = np.zeros((nx + 1, ny + 1, nz + 1, 3), bool)
    fixed[0, :, :, 0] = True
    fixed[0, 0, 0, 1:] = True  # u_y = u_z = 0 in one corner
    fixed[0, ny, 0, 2] = True  # u_z = 0 in a second corner on the y axis: no rotation about x
    f = np.zeros(fixed.shape)
    for j, k in np.ndindex(ny, nz):
        f[nx, j:j + 2, k:k + 2, 0] += sigma * H * H / 4  # consistent nodal forces of a uniform face traction
    return np.ones((nx, ny, nz), bool), fixed, f


def test_element_matrix_is_symmetric_with_exactly_six_rigid_modes():
    K = fem.element_stiffness(NU)
    w = np.linalg.eigvalsh(K)
    assert np.allclose(K, K.T) and (np.abs(w[:6]) < 1e-12).all() and (w[6:] > 1e-3).all()


def test_uniform_tension_matches_the_analytic_solution():
    """Linear displacement fields lie in the trilinear space, so the FE answer must be exact: u = (s x / E, -nu s y / E, -nu s z / E)."""
    sigma = 1e6
    solid, fixed, f = tension_block(sigma)
    u = fem.solve(solid, fixed, f, E, NU, H)
    X, Y, Z = np.indices(fixed.shape[:3]) * H
    exact = np.stack([sigma * X / E, -NU * sigma * Y / E, -NU * sigma * Z / E], -1)
    assert np.abs(u - exact).max() < 1e-9 * np.abs(exact).max()
    assert np.isclose((f * u).sum(), sigma ** 2 * solid.sum() * H ** 3 / E, rtol=1e-9)  # compliance = s^2 V / E
    K = E * H * fem.assemble(solid, NU)
    assert np.isclose(u.ravel() @ (K @ u.ravel()), (f * u).sum(), rtol=1e-9)  # f.u = u.K.u: the stored stiffness and the solver agree


def test_void_voxels_do_not_change_the_answer():
    """The same block inside a shell of void voxels: only E_VOID of the stiffness is added, and the floating void nodes must not break the solve."""
    solid, fixed, f = tension_block(1e6)
    u = fem.solve(solid, fixed, f, E, NU, H)
    pad = [(1, 0), (0, 2), (3, 1)]  # a different shell on every axis and side, so a mix-up of the x / y / z voxel order cannot cancel out
    big = fem.solve(np.pad(solid, pad), np.pad(fixed, pad + [(0, 0)]), np.pad(f, pad + [(0, 0)]), E, NU, H)
    nx, ny, nz = solid.shape
    assert np.abs(big[1:nx + 2, :ny + 1, 3:nz + 4] - u).max() < 1e-6 * np.abs(u).max()


def tip_deflection(n, P=1000.0):
    """Tip deflection (m) of an 8 : 1 : 1 cantilever, 1 cm thick, n voxels through the thickness, clamped at x = 0, end load P in z."""
    nx, ny, nz = 8 * n, n, n
    fixed = np.zeros((nx + 1, ny + 1, nz + 1, 3), bool)
    fixed[0] = True
    f = np.zeros(fixed.shape)
    f[nx, :, :, 2] = P / ((ny + 1) * (nz + 1))
    return fem.solve(np.ones((nx, ny, nz), bool), fixed, f, E, NU, 0.01 / n)[nx, :, :, 2].mean()


def test_cantilever_converges_to_beam_theory():
    """Bending is where trilinear voxels are too stiff: with 1 voxel through the thickness the tip moves only 64% of the beam-theory value."""
    L, t, P = 0.08, 0.01, 1000.0
    beam = P * L ** 3 / (3 * E * t ** 4 / 12) + P * L / (5 / 6 * E / (2 * (1 + NU)) * t * t)  # bending + shear (Timoshenko)
    err = [abs(tip_deflection(n) / beam - 1) for n in (1, 2, 4, 8)]
    assert err[0] > 0.3 and err[0] > err[1] > err[2] > err[3] and err[3] < 0.03, err


def test_body_force_and_supports_are_converted_per_voxel():
    h, F = 0.0025, 1e10  # F in N/m^3, so one voxel carries F * h^3 = 156.25 N
    f = fem.nodal_loads((3, 2, 2), np.array([[0, 0, 0], [1, 0, 0]]), np.array([[F, 0, 0], [F, 0, 0]]), h)
    assert np.isclose(f.sum(), 2 * 156.25) and np.isclose(f[0, 0, 0, 0], 156.25 / 8)
    assert np.isclose(f[1, 0, 0, 0], 2 * 156.25 / 8)  # a node shared by two loaded voxels gets two eighths
    assert f[..., 1:].sum() == 0
    flags = np.zeros((3, 2, 2, 3), bool)
    flags[1, 1, 0, 0] = True  # one voxel fixed in x only
    fixed = fem.to_nodes(flags)
    assert fixed.sum() == 8 and fixed[..., 0].sum() == 8 and fixed[1:3, 1:3, 0:2, 0].all()


def test_more_material_is_never_less_stiff_and_loads_are_round_numbers():
    """Real sample: filling every free voxel can only lower the compliance; the loads after F * h^3 are 1500 - 2000 N."""
    if not (A.DATA / "test.npz").exists():
        return print("skipped: run src/prepare.py first")
    d = A.load("test")
    for i in (0, 123):
        fixed, f = fem.sample_problem(d, i)
        assert 1499 < np.linalg.norm(f.sum((0, 1, 2))) < 2001
        c_ref = fem.mechanics(d["density"][i], fixed, f, d["info"])[0]
        c_full = fem.mechanics(d["density"][i] | (d["design"][i] == A.FREE), fixed, f, d["info"])[0]
        assert 0 < c_full < c_ref, (i, c_full, c_ref)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
