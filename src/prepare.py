"""Parse the raw SELTO CSVs (disc_simple, plus the test splits of the other subsets) into compact arrays.

Usage: python -I src/prepare.py data/raw data/processed
Raw files are untrusted data: they are only parsed as numbers here, never executed or unpickled.

CSV layout per voxel row (no header): x, y, z, design_space, dirichlet_x, dirichlet_y, dirichlet_z,
force_x, force_y, force_z, density. Rows are x-major with z fastest, so a plain reshape gives [x, y, z].
design_space: -1 free (optimised), 0 void (must stay empty), 1 solid (must stay filled); tests/test_core.py checks this reading.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS = {"train": "disc_simple_train", "test": "disc_simple_test", "disc_complex": "disc_complex_test",
           "sphere_simple": "sphere_simple_test", "sphere_complex": "sphere_complex_test"}  # processed name -> raw folder


def load_split(folder: Path) -> dict:
    n = len(list(folder.glob("*_info.csv")))
    assert n and all((folder / f"{i}.csv").exists() for i in range(n)), f"{folder}: samples must be numbered 0..n-1"
    SHAPE = tuple(int(v) + 1 for v in pd.read_csv(folder / "0.csv", header=None).to_numpy()[:, :3].max(0))  # disc 39x39x4, sphere 39x39x21
    COORDS = np.stack(np.meshgrid(*map(np.arange, SHAPE), indexing="ij"), -1).reshape(-1, 3)
    design = np.empty((n, *SHAPE), np.int8)
    density = np.empty((n, *SHAPE), np.uint8)
    dirichlet = info = None
    f_idx, f_val = [], []  # sparse list of loaded voxels: (sample, x, y, z) and (fx, fy, fz)
    for i in range(n):
        a = pd.read_csv(folder / f"{i}.csv", header=None).to_numpy()
        assert a.shape == (len(COORDS), 11) and (a[:, :3] == COORDS).all(), f"{folder}/{i}.csv: unexpected layout"
        assert np.isin(a[:, 3], (-1, 0, 1)).all() and np.isin(a[:, 10], (0, 1)).all(), f"{folder}/{i}.csv: unexpected values"
        design[i] = a[:, 3].reshape(SHAPE)
        density[i] = a[:, 10].reshape(SHAPE)
        d = a[:, 4:7].astype(np.uint8).reshape(*SHAPE, 3)
        p = np.loadtxt(folder / f"{i}_info.csv", delimiter=",")  # E, Poisson ratio, yield stress, voxel size
        if dirichlet is None:
            dirichlet, info = d, p
        # supports and material constants are shared by all samples, so they are stored once
        assert (d == dirichlet).all() and (p == info).all(), f"{folder}/{i}: supports/material differ from sample 0"
        rows = np.flatnonzero((a[:, 7:10] != 0).any(1))
        f_idx.append(np.c_[np.full(len(rows), i), a[rows, :3]].astype(np.int16))
        f_val.append(a[rows, 7:10])
    return dict(design=design, density=density, dirichlet=dirichlet, info=info,
                force_idx=np.concatenate(f_idx), force_val=np.concatenate(f_val))


if __name__ == "__main__":
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for name, folder in OUTPUTS.items():
        if (raw / folder).exists():
            data = load_split(raw / folder)
            np.savez_compressed(out / f"{name}.npz", **data)
            print(name, {k: v.shape for k, v in data.items()})
