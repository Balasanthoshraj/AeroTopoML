"""Self-checks for the logic that is easy to get silently wrong: symmetry augmentation, voxel meshing, post-processing, metrics.

  python tests/test_core.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import aerotopoml as A  # noqa: E402
import viz3d  # noqa: E402


def test_augment_keeps_grid_and_features_consistent():
    dens = np.zeros((1, 39, 39, 4), bool)
    dens[0, 5, 10, 1] = True  # one marked voxel where the load acts
    d = dict(density=dens, centroid=np.array([[5.0, 10.0, 1.0]]), force=np.array([[1.0, 0.5, 0.0]]))
    out = A.augment(d)
    assert out["density"].shape == (8, 39, 39, 4) and out["x"].shape == (8, 2 + 6 * A.HARMONICS)
    for g in range(8):  # the load position feature must follow the voxel through every rotation / mirror
        x, y, _ = np.argwhere(out["density"][g])[0]
        assert np.allclose(out["x"][g, :2], [(x - A.AXIS) / A.AXIS, (y - A.AXIS) / A.AXIS], atol=1e-6), g
    cos_rel, sin_rel = out["x"][:, 8], out["x"][:, 17]  # load direction relative to the radial direction
    assert np.allclose(cos_rel, cos_rel[0], atol=1e-5)  # unchanged by rotations and mirrors
    assert np.allclose(sin_rel[:4], sin_rel[0], atol=1e-5) and np.allclose(sin_rel[4:], -sin_rel[0], atol=1e-5)  # sign flips when mirrored


def test_voxel_mesh_is_closed_and_outward():
    lab = np.zeros((4, 4, 4), np.int8)
    lab[1, 1, 1] = 1
    v, q, c = viz3d.voxel_mesh(lab)
    assert len(v) == 8 and len(q) == 6 and set(c) == {1}
    for quad in q:  # every face normal points away from the cube centre
        p = v[quad].astype(float)
        assert np.cross(p[1] - p[0], p[2] - p[0]) @ (p.mean(0) - v.mean(0)) > 0
    lab[2, 1, 1] = 1  # a neighbour hides the two touching faces
    assert len(viz3d.voxel_mesh(lab)[1]) == 10


def test_post_processing_and_metrics():
    b = np.zeros(A.SHAPE, bool)
    b[0:5, 3, 1] = True
    load, support = np.zeros(A.SHAPE, bool), np.zeros(A.SHAPE, bool)
    load[0, 3, 1], support[4, 3, 1] = True, True
    assert A.connected(b, load, support) == (1, True)
    b[2, 3, 1] = False  # break the load path
    assert A.connected(b, load, support) == (2, False)
    design = np.full(A.SHAPE, A.FREE, np.int8)
    design[0, 0, 0], design[1, 1, 1] = A.SOLID, A.VOID
    out = A.binarize(np.full(A.SHAPE, 0.9), design)
    assert out[0, 0, 0] and not out[1, 1, 1] and out[2, 2, 2]
    a = np.zeros((1, 4, 4, 4), bool)
    a[0, 0] = True
    assert A.overlap(a, a)[0][0] == 1 and A.overlap(a, ~a)[0][0] == 0


def test_reference_structures_obey_the_design_space_codes():
    """-1 free / 0 void / 1 solid: the supplied optimised structures must already satisfy them in every sample."""
    if not (A.DATA / "test.npz").exists():
        return print("skipped: run src/prepare.py first")
    for split in ("train", "test"):
        d = A.load(split)
        assert (A.binarize(d["density"].astype(float), d["design"]) == d["density"]).all(), split


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
