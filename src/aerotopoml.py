"""AeroTopoML core: load-condition features, PCA + MLP topology predictor, post-processing and metrics.

Data comes from prepare.py (data/processed/{train,test}.npz). Everything here is CPU-only.
"""
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "processed", ROOT / "outputs"
SHAPE = (39, 39, 4)
FREE, VOID, SOLID = -1, 0, 1  # design_space codes: optimised / must stay empty / must stay filled
AXIS = 19.0  # x and y index of the disc axis
HARMONICS = 3  # angle harmonics in the load features (1 = plain cos / sin)
CONN = ndimage.generate_binary_structure(3, 1)  # face-connected voxels


# ---------------------------------------------------------------- data and features
def load(split: str) -> dict:
    """Load a processed split and add per-sample masks and the compact load-condition features d['x']."""
    with np.load(DATA / f"{split}.npz") as z:
        d = {k: z[k] for k in z.files}
    n = len(d["density"])
    d["density"] = d["density"].astype(bool)
    d["load"] = np.zeros((n, *SHAPE), bool)
    d["load"][tuple(d["force_idx"].T.astype(int))] = True
    d["support"] = d["dirichlet"].any(-1)[None] & (d["design"] == SOLID)
    d["centroid"], d["force"] = load_summary(d["force_idx"], d["force_val"], n)
    d["x"] = features(d["centroid"], d["force"])
    return d


def load_summary(force_idx, force_val, n):
    """Force-weighted centroid (x, y, z in voxels) and total force vector of the loaded voxels, per sample."""
    s, w = force_idx[:, 0].astype(int), np.linalg.norm(force_val, axis=1)

    def per_sample(a):
        return np.stack([np.bincount(s, weights=a[:, j], minlength=n) for j in range(3)], 1)

    centroid = per_sample(force_idx[:, 1:] * w[:, None]) / np.bincount(s, weights=w, minlength=n)[:, None]
    return centroid, per_sample(force_val)


def features(centroid, force, harmonics=HARMONICS):
    """Load descriptors, 2 + 6 * harmonics numbers per sample: the load position (x, y) and cos / sin of the harmonics of three angles
    (load direction, load position on the rim, and direction relative to the radial direction).

    Fz is zero in every disc_simple sample, and E / Poisson ratio / yield stress / supports are constant, so none of those are used.
    """
    theta = np.arctan2(force[:, 1], force[:, 0])
    phi = np.arctan2(centroid[:, 1] - AXIS, centroid[:, 0] - AXIS)
    angles = [m * a for a in (theta, phi, theta - phi) for m in range(1, harmonics + 1)]
    return np.stack([(centroid[:, 0] - AXIS) / AXIS, (centroid[:, 1] - AXIS) / AXIS, *np.cos(angles), *np.sin(angles)], 1).astype(np.float32)


def augment(d):
    """Training set expanded with the 8 symmetries of the square about the disc axis (90-degree rotations and mirror images).

    The disc, its hub and the supports are symmetric, so a rotated or mirrored problem has the rotated or mirrored optimum.
    Only what fit() needs is returned: densities and load features.
    """
    dens, x = [], []
    for flip in (False, True):
        g, c, f = d["density"], d["centroid"].copy(), d["force"].copy()
        if flip:
            g, c[:, 0], f[:, 0] = g[:, ::-1], 2 * AXIS - c[:, 0], -f[:, 0]
        for _ in range(4):
            dens.append(g)
            x.append(features(c, f))
            g = np.rot90(g, axes=(1, 2))
            c[:, :2] = np.c_[2 * AXIS - c[:, 1], c[:, 0]]
            f[:, :2] = np.c_[-f[:, 1], f[:, 0]]
    return dict(density=np.concatenate(dens), x=np.concatenate(x))


def split(n, frac=0.2, seed=0):
    """Fixed random train / internal-validation split of the supplied training set."""
    p = np.random.default_rng(seed).permutation(n)
    return np.sort(p[int(n * frac):]), np.sort(p[:int(n * frac)])


PER_SAMPLE = ("density", "design", "load", "support", "centroid", "force", "x")


def take(d, idx):
    """Subset the per-sample arrays of a loaded split (the sparse force tables are not carried over)."""
    return {k: d[k][idx] for k in PER_SAMPLE} | {k: d[k] for k in ("dirichlet", "info")}


# ---------------------------------------------------------------- model
def train_mlp(Xtr, Ytr, Xva, Yva, hidden=(128, 128), lr=2e-3, wd=1e-5, bs=64, max_epochs=1500, patience=80, seed=0):
    """Adam + MSE (learning rate halved when the validation loss stalls), early stopping on the validation loss.
    Returns the best model and the (train, val) loss history."""
    torch.manual_seed(seed)
    Xtr, Ytr, Xva, Yva = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))
    dims = [Xtr.shape[1], *hidden]
    layers = [m for a, b in zip(dims, dims[1:]) for m in (torch.nn.Linear(a, b), torch.nn.ReLU())]
    model = torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], Ytr.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience // 3, min_lr=1e-5)
    mse = torch.nn.functional.mse_loss
    best, best_state, wait, hist = float("inf"), None, 0, []
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            mse(model(Xtr[b]), Ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hist.append((mse(model(Xtr), Ytr).item(), mse(model(Xva), Yva).item()))
        sched.step(hist[-1][1])
        if hist[-1][1] < best - 1e-7:
            best, wait, best_state = hist[-1][1], 0, {k: v.clone() for k, v in model.state_dict().items()}
        elif (wait := wait + 1) >= patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), np.array(hist)


class Pipeline:
    """Load features -> standardise -> regressor -> PCA coefficients -> inverse PCA -> density field (N, 39, 39, 4)."""

    def __init__(self, pca, mu, sd, scale, reg, history=None):
        self.pca, self.mu, self.sd, self.scale, self.reg, self.history = pca, mu, sd, scale, reg, history

    def coef(self, x):
        z = ((x - self.mu) / self.sd).astype(np.float32)
        if isinstance(self.reg, torch.nn.Module):
            with torch.no_grad():
                return self.reg(torch.from_numpy(z)).numpy() * self.scale
        return self.reg.predict(z) * self.scale

    def density(self, x):
        return self.pca.inverse_transform(self.coef(x)).reshape(-1, *SHAPE)


class Basis:
    """Inference-only stand-in for a fitted sklearn PCA (reconstruction is mean + coefficients @ components)."""

    def __init__(self, components, mean):
        self.components_, self.mean_ = components, mean

    def inverse_transform(self, c):
        return c @ self.components_ + self.mean_


def save_pipeline(pipe, config, folder=OUT / "model"):
    """Store the fitted MLP pipeline as plain arrays + torch weights (no pickle) and its post-processing config."""
    import json
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "pipeline.npz", components=pipe.pca.components_, mean=pipe.pca.mean_, mu=pipe.mu, sd=pipe.sd, scale=pipe.scale)
    torch.save(pipe.reg.state_dict(), folder / "mlp.pt")
    (folder / "config.json").write_text(json.dumps(config, indent=2))


def load_pipeline(folder=OUT / "model"):
    """Inverse of save_pipeline: returns (Pipeline, config)."""
    import json
    with np.load(folder / "pipeline.npz") as z:
        a = {k: z[k] for k in z.files}
    sd = torch.load(folder / "mlp.pt", weights_only=True)
    widths = [v.shape for k, v in sd.items() if k.endswith("weight")]
    layers = [m for w in widths[:-1] for m in (torch.nn.Linear(w[1], w[0]), torch.nn.ReLU())]
    mlp = torch.nn.Sequential(*layers, torch.nn.Linear(widths[-1][1], widths[-1][0]))
    mlp.load_state_dict(sd)
    return Pipeline(Basis(a["components"], a["mean"]), a["mu"], a["sd"], float(a["scale"]), mlp.eval()), json.loads((folder / "config.json").read_text())


def fit(tr, va, k, kind="mlp", **mlp_kw):
    """Fit PCA on the training structures only, then a regressor from load features to the k PCA coefficients."""
    Ytr = tr["density"].reshape(len(tr["density"]), -1).astype(np.float32)
    pca = PCA(k, svd_solver="full", random_state=0).fit(Ytr)
    mu, sd = tr["x"].mean(0), tr["x"].std(0) + 1e-8
    ctr = pca.transform(Ytr)
    cva = pca.transform(va["density"].reshape(len(va["density"]), -1).astype(np.float32))
    scale = ctr.std()  # one global factor, so the loss keeps voxel-space weighting of the components
    ztr, zva = ((tr["x"] - mu) / sd).astype(np.float32), ((va["x"] - mu) / sd).astype(np.float32)
    if kind == "linear":
        return Pipeline(pca, mu, sd, scale, LinearRegression().fit(ztr, ctr / scale))
    reg, hist = train_mlp(ztr, (ctr / scale).astype(np.float32), zva, (cva / scale).astype(np.float32), **mlp_kw)
    return Pipeline(pca, mu, sd, scale, reg, hist)


def pca_only(tr, k):
    """PCA reconstruction ceiling: what remains after compressing the reference structures to k components."""
    Y = tr["density"].reshape(len(tr["density"]), -1).astype(np.float32)
    return PCA(k, svd_solver="full", random_state=0).fit(Y)


# ---------------------------------------------------------------- post-processing and metrics
def binarize(dens, design, tau=0.5):
    """Threshold a density field, then enforce the mandatory design-space regions."""
    b = dens > tau
    b[design == VOID] = False
    b[design == SOLID] = True
    return b


def connected(b, load, support):
    """Number of material components, and whether one component touches both the load patch and the supports."""
    lab, n = ndimage.label(b, CONN)
    return n, bool((set(np.unique(lab[load])) & set(np.unique(lab[support]))) - {0})


def structures(dens, d, tau=0.5, repair=False, clean=False, floor=0.15):
    """Binary structures from density fields. repair: lower the threshold until load and supports are joined;
    clean: drop material that is not connected to the load or the supports."""
    out = binarize(dens, d["design"], tau)
    if not (repair or clean):
        return out
    for i in range(len(out)):
        t = tau
        while repair and not connected(out[i], d["load"][i], d["support"][i])[1] and t > floor:
            t -= 0.05
            out[i] = binarize(dens[i], d["design"][i], t)
        if clean:
            lab, _ = ndimage.label(out[i], CONN)
            keep = np.unique(np.r_[lab[d["load"][i]], lab[d["support"][i]]])
            out[i] &= np.isin(lab, keep[keep > 0])
    return out


def overlap(pred, ref, mask=None):
    """Per-sample Dice and IoU of boolean voxel grids, optionally restricted to a mask."""
    if mask is not None:
        pred, ref = pred & mask, ref & mask
    ax = tuple(range(1, pred.ndim))
    inter, p, r = (pred & ref).sum(ax), pred.sum(ax), ref.sum(ax)
    dice = np.where(p + r == 0, 1.0, 2 * inter / np.maximum(p + r, 1))
    return dice, np.where(p + r == 0, 1.0, inter / np.maximum(p + r - inter, 1))


def evaluate(dens, d, **post):
    """Per-sample metrics of predicted density fields against the reference structures of split d."""
    pred, ref, free = structures(dens, d, **post), d["density"], d["design"] == FREE
    dice, iou = overlap(pred, ref)
    dice_f, iou_f = overlap(pred, ref, free)
    raw, fixed = dens > post.get("tau", 0.5), d["design"] != FREE
    wrong = ((raw & (d["design"] == VOID)) | (~raw & (d["design"] == SOLID))).sum((1, 2, 3))
    conn = [connected(p, ld, sp) for p, ld, sp in zip(pred, d["load"], d["support"])]
    return dict(dice=dice, iou=iou, dice_free=dice_f, iou_free=iou_f,
                vol_err=np.abs(pred.sum((1, 2, 3)) - ref.sum((1, 2, 3))) / ref.sum((1, 2, 3)),
                violation=wrong / fixed.sum((1, 2, 3)),
                components=np.array([c[0] for c in conn]), joined=np.array([c[1] for c in conn]))


def summary(m):
    """Mean of each per-sample metric."""
    return {k: float(np.mean(v)) for k, v in m.items()}


def inference_ms(pipe, d, n=100, **post):
    """Median wall-clock time to turn one set of load conditions into a binary structure (features -> structure)."""
    t = []
    for i in range(min(n, len(d["density"]))):
        one = take(d, [i])
        t0 = time.perf_counter()
        structures(pipe.density(one["x"]), one, **post)
        t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))
