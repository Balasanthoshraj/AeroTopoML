"""Self-checks of the part optimiser: element stiffness, sensitivities, a short optimisation, the parts library, the viewer surface.

  python tests/test_topopt.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import objects as O  # noqa: E402
import topopt as T  # noqa: E402


def test_element_stiffness_has_exactly_the_rigid_body_modes():
    Ke = T.hex8_stiffness()
    assert np.allclose(Ke, Ke.T)
    assert np.linalg.matrix_rank(Ke, tol=1e-9) == 18  # 24 dofs minus 3 translations and 3 rotations
    c = T.CORNERS / 2  # corners of the unit cube, centred
    for rigid in (np.tile([1.0, 0, 0], 8), np.c_[-c[:, 1], c[:, 0], np.zeros(8)].ravel(), np.c_[np.zeros(8), -c[:, 2], c[:, 1]].ravel()):
        assert np.abs(Ke @ rigid).max() < 1e-12


def test_sensitivities_match_finite_differences():
    design = np.full((5, 3, 3), T.FREE, np.int8)
    fixed, force = np.zeros((5, 3, 3, 3), bool), np.zeros((5, 3, 3, 3))
    design[0], fixed[0] = T.SOLID, True
    design[-1, 1, :], force[-1, 1, :, 2] = T.SOLID, -1.0
    m = T.Model(T.Problem("beam", design, fixed, force))
    x, p = np.random.default_rng(0).uniform(0.3, 1, m.n), 3.0
    compliance = lambda x: m.f @ m.solve(T.EMIN + (1 - T.EMIN) * x**p)
    analytic = -p * (1 - T.EMIN) * x ** (p - 1) * m.energy(m.solve(T.EMIN + (1 - T.EMIN) * x**p))
    for e in np.argsort(analytic)[:3]:  # the elements that carry the most energy
        up, dn = x.copy(), x.copy()
        up[e] += 1e-6
        dn[e] -= 1e-6
        assert abs((compliance(up) - compliance(dn)) / 2e-6 / analytic[e] - 1) < 1e-4
    m.close()


def test_optimiser_reaches_the_mass_target_and_lowers_compliance():
    p = O.build("bracket", "Fast")
    res = T.optimize(p, iterations=20)
    n = int((p.design != T.VOID).sum())
    assert res.history[-1]["compliance"] < 0.6 * res.history[0]["compliance"]
    assert abs(res.structure.sum() - p.keep * n) <= 0.02 * n  # exact mass, minus any floating pieces that were dropped
    assert 0 < res.metrics["stiffness_retained"] <= 1
    lab, _ = ndimage.label(res.structure, ndimage.generate_binary_structure(3, 1))
    assert set(np.unique(lab[res.structure])) == set(np.unique(lab[p.fixed.any(-1) & res.structure]))  # every piece touches a support


def test_every_part_builds_and_validates_at_every_detail():
    for key, spec in O.CATALOG.items():
        if not spec.dataset:  # the dataset entries are built from data, not from a shape
            for detail in O.DETAIL:
                T.validate(O.build(key, detail))


def test_every_dataset_example_is_a_valid_problem_with_the_reference_mass():
    import aerotopoml as A
    for spec in O.CATALOG.values():
        if not spec.dataset or not (A.DATA / f"{spec.dataset}.npz").exists():
            continue
        with np.load(A.DATA / f"{spec.dataset}.npz") as z:
            d = {k: z[k] for k in z.files}
        for i in (0, len(d["density"]) - 1):
            p = O.dataset_problem(d, i, spec.label)
            T.validate(p)
            assert abs(np.linalg.norm(p.force, axis=-1).sum() - 3_000) < 1e-6  # loads are scaled to a total of 3 kN
            assert abs(p.keep * (p.design != T.VOID).sum() - d["density"][i].sum()) < 1e-6  # the reference's mass


def test_smooth_surface_of_a_block_is_closed():
    import viz3d
    s = np.zeros((6, 6, 6), bool)
    s[1:5, 1:5, 1:5] = True
    poly = viz3d.smooth_surface(s)
    assert poly.n_cells > 0 and poly.n_open_edges == 0


def test_physics_optimiser_agrees_with_the_dataset_reference():
    """On a SELTO test sample this solver lands on the OptiStruct reference: Dice above 0.85 on the voxels it decides."""
    import aerotopoml as A
    if not (A.DATA / "test.npz").exists():
        return print("skipped: run src/prepare.py first")
    d = A.load("test")
    p = O.dataset_problem(d, 82, "Car wheel")
    res = T.optimize(p, iterations=50)
    free, ref = p.design == T.FREE, d["density"][82]
    dice = 2 * (res.structure & ref & free).sum() / ((res.structure & free).sum() + (ref & free).sum())
    assert dice > 0.85, dice


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
