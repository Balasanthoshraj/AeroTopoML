"""Experiments from the proposal on SELTO disc_simple.

  python src/run_experiments.py            full run (tens of minutes on CPU)
  python src/run_experiments.py --quick    smoke test with a tiny grid, writes to outputs_quick/

PCA reconstruction, linear regression + PCA, MLP + PCA, PCA dimensions, MLP sizes and symmetry augmentation are compared on the
internal validation split (same split and metrics everywhere). The selected model is then evaluated once on the supplied test
split. Writes results.csv, metrics.json, test_per_sample.csv, figures/ and model/ under outputs/.
"""
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import aerotopoml as A

KS = (32, 64, 128, 256)
SEEDS = (0, 1, 2)  # MLP initialisation seeds
SIZES = {"2x32": (32, 32), "2x64": (64, 64), "2x128": (128, 128), "2x256": (256, 256), "3x256": (256, 256, 256)}
AUG_SEEDS = (0, 1)  # augmented training is about 10x slower
QUICK = "--quick" in sys.argv
FAST = dict(max_epochs=30) if QUICK else {}
if QUICK:
    KS, SEEDS, AUG_SEEDS, SIZES = (32, 64), (0,), (0,), {"2x32": (32, 32), "2x64": (64, 64)}
rows = []


def log(experiment, detail, split, dens, d, seed=None, post=None, **extra):
    m = A.summary(A.evaluate(dens, d, **(post or {})))
    rows.append(dict(experiment=experiment, detail=detail, split=split, seed=seed, **m, **extra))
    print(f"{experiment:22s} {detail:36s} {split:4s} dice {m['dice']:.3f} | free voxels: dice {m['dice_free']:.3f} iou {m['iou_free']:.3f}"
          f" | joined {m['joined']:.2f}", flush=True)
    return m


def best_of(df, experiment, tol=0.0):
    """Smallest (first listed) setting whose mean validation Dice on free voxels is within tol of the best one."""
    q = df[(df.experiment == experiment) & (df.split == "val")].groupby("detail", sort=False).dice_free.mean()
    return next(n for n, v in q.items() if v >= q.max() - tol)


def plot_pca_and_models(df, ev, fig_dir):
    """Figures 1 (PCA) and 2 (linear vs MLP by PCA dimension) from the results table and the cumulative explained variance ev."""
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    val = df[df.split == "val"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(np.arange(1, len(ev) + 1), ev)
    for k in KS:
        ax[0].axvline(k, ls=":", c="grey")
    ax[0].set(xlabel="principal components", ylabel="cumulative explained variance", title="PCA of the reference structures")
    for off, sp in ((-0.2, "val"), (0.2, "test")):
        q = df[(df.experiment == "PCA reconstruction") & (df.split == sp)]
        ax[1].bar(np.arange(len(KS)) + off, q.dice_free, 0.4, label=sp)
    ax[1].set(xticks=range(len(KS)), xticklabels=[f"k={k}" for k in KS], ylim=(0, 1.15), yticks=np.arange(0, 1.01, 0.2),
              ylabel="Dice on free voxels", title="PCA reconstruction ceiling")
    ax[1].legend(ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(fig_dir / "1_pca.png")

    lin_v = val[val.experiment == "linear + PCA"].dice_free.values
    mlp_v = val[val.experiment == "MLP + PCA"].groupby("detail", sort=False).dice_free.agg(["mean", "std"])
    ceil = val[val.experiment == "PCA reconstruction"].dice_free.values
    x = np.arange(len(KS))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - 0.2, lin_v, 0.4, label="linear + PCA")
    ax.bar(x + 0.2, mlp_v["mean"], 0.4, yerr=mlp_v["std"], capsize=3, label="MLP 2x128 + PCA (mean, std of 3 seeds)")
    ax.plot(x, ceil, "k_", ms=22, mew=2, label="PCA reconstruction ceiling")
    ax.set(xticks=x, xticklabels=[f"k={k}" for k in KS], ylabel="Dice on free voxels (validation)", ylim=(0, 1.3), yticks=np.arange(0, 1.01, 0.2),
           title="Linear vs MLP, by PCA dimension")
    ax.legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "2_models_by_k.png")


def main():
    t_start = time.time()
    out = A.ROOT / "outputs_quick" if QUICK else A.OUT
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    train, test = A.load("train"), A.load("test")
    itr, iva = A.split(len(train["density"]))
    tr, va = A.take(train, itr), A.take(train, iva)
    print(f"split: {len(itr)} train / {len(iva)} internal validation / {len(test['density'])} test (supplied)")

    # 0. trivial references, so the Dice numbers can be read in context
    for name, d in (("val", va), ("test", test)):
        log("reference", "fixed regions only", name, np.zeros(d["density"].shape, np.float32), d)
        j = NearestNeighbors(n_neighbors=1).fit(tr["x"]).kneighbors(d["x"])[1][:, 0]
        log("reference", "1-NN on load features", name, tr["density"][j].astype(np.float32), d)

    # 1. PCA reconstruction: information lost by compression alone
    pcas = {k: A.pca_only(tr, k) for k in KS}
    for k, pca in pcas.items():
        for name, d in (("val", va), ("test", test)):
            Y = d["density"].reshape(len(d["density"]), -1).astype(np.float32)
            log("PCA reconstruction", f"k={k}", name, pca.inverse_transform(pca.transform(Y)).reshape(-1, *A.SHAPE), d,
                explained_var=float(pca.explained_variance_ratio_.sum()))

    # 2-3. linear regression + PCA versus MLP + PCA, for each PCA dimension
    fits = {}
    for k in KS:
        fits["linear", k] = A.fit(tr, va, k, "linear")
        log("linear + PCA", f"k={k}", "val", fits["linear", k].density(va["x"]), va)
        for s in SEEDS:
            fits["mlp", k, s] = A.fit(tr, va, k, "mlp", seed=s, **FAST)
            log("MLP + PCA", f"k={k}", "val", fits["mlp", k, s].density(va["x"]), va, seed=s, epochs=len(fits["mlp", k, s].history))
    best_k = int(best_of(pd.DataFrame(rows), "MLP + PCA", tol=0.005).split("=")[1])
    print(f"-> PCA dimension (smallest within 0.005 of the best MLP score): k={best_k}")

    # 4. MLP sizes at the best k
    for name, hidden in SIZES.items():
        for s in SEEDS:
            p = fits["mlp", best_k, s] if hidden == (128, 128) else A.fit(tr, va, best_k, "mlp", hidden=hidden, seed=s, **FAST)
            fits["size", name, s] = p
            h = p.history
            e = h[:, 1].argmin()
            log("MLP size", name, "val", p.density(va["x"]), va, seed=s, train_loss=h[e, 0], val_loss=h[e, 1], epochs=len(h))
    best_size = best_of(pd.DataFrame(rows), "MLP size", tol=0.003)
    print(f"-> MLP size (smallest within 0.003 of the best): {best_size}")

    # 4b. effect of the angle harmonics in the input features
    tr1, va1 = (dict(d, x=A.features(d["centroid"], d["force"], 1)) for d in (tr, va))
    for s in SEEDS:
        log("input features", "plain cos/sin (8 numbers)", "val", A.fit(tr1, va1, best_k, "mlp", seed=s, **FAST).density(va1["x"]), va, seed=s)
        log("input features", "with harmonics (20 numbers)", "val", fits["mlp", best_k, s].density(va["x"]), va, seed=s)

    # 5. symmetry augmentation (rotations and mirror images of the training problems)
    aug = A.augment(A.take(tr, np.arange(100)) if QUICK else tr)
    for s in AUG_SEEDS:
        fits["aug", s] = A.fit(aug, va, best_k, "mlp", hidden=SIZES[best_size], seed=s, **(dict(bs=256, max_epochs=600) | FAST))
        log("symmetry augmentation", f"{best_size}, 8x training data", "val", fits["aug", s].density(va["x"]), va, seed=s,
            epochs=len(fits["aug", s].history))
    df = pd.DataFrame(rows)
    use_aug = df[df.experiment == "symmetry augmentation"].dice_free.mean() > df[(df.experiment == "MLP size") & (df.detail == best_size)].dice_free.mean()
    print(f"-> symmetry augmentation {'helps' if use_aug else 'does not help'} on validation")
    cands = [fits["aug", s] for s in AUG_SEEDS] if use_aug else [fits["size", best_size, s] for s in SEEDS]
    final = cands[int(np.argmax([A.summary(A.evaluate(c.density(va["x"]), va))["dice_free"] for c in cands]))]

    # 6. thresholding and structural post-processing, tuned on the validation split
    dens_va = final.density(va["x"])
    taus = np.round(np.arange(0.2, 0.71, 0.05), 2)
    tau = float(taus[int(np.argmax([log("threshold", f"tau={t:.2f}", "val", dens_va, va, post=dict(tau=float(t)))["dice_free"] for t in taus]))])
    variants = {"threshold only": {}, "+ reconnect load and supports": dict(repair=True), "+ remove floating material": dict(clean=True),
                "+ both": dict(repair=True, clean=True)}
    res = {n: log("post-processing", n, "val", dens_va, va, post=dict(tau=tau, **kw)) for n, kw in variants.items()}
    top = max(r["dice_free"] for r in res.values())
    choice = max((n for n, r in res.items() if r["dice_free"] >= top - 0.01), key=lambda n: (res[n]["joined"], res[n]["dice_free"]))
    post = dict(tau=tau, repair=False, clean=False) | variants[choice]
    print(f"-> post-processing: {choice}, {post}")

    # 7. final evaluation on the supplied test split (used once)
    dens_te = final.density(test["x"])
    lin = fits["linear", best_k]
    log("final", "linear + PCA, tau 0.5", "test", lin.density(test["x"]), test)
    log("final", "linear + PCA, tuned post-processing", "test", lin.density(test["x"]), test, post=post)
    log("final", "MLP + PCA, tau 0.5", "test", dens_te, test)
    m_te = log("final", "MLP + PCA, tuned post-processing", "test", dens_te, test, post=post)
    ms_single = A.inference_ms(final, test, **post)
    t0 = time.perf_counter()
    A.structures(final.density(test["x"]), test, **post)
    ms_batch = 1000 * (time.perf_counter() - t0) / len(test["density"])

    per = A.evaluate(dens_te, test, **post)
    pred = A.structures(dens_te, test, **post)
    ps = pd.DataFrame(per | dict(
        ref_voxels=test["density"].sum((1, 2, 3)), pred_voxels=pred.sum((1, 2, 3)),
        direction_deg=np.degrees(np.arctan2(test["force"][:, 1], test["force"][:, 0])),
        position_deg=np.degrees(np.arctan2(test["centroid"][:, 1] - A.AXIS, test["centroid"][:, 0] - A.AXIS))))
    ps.index.name = "test_sample"
    ps.round(4).to_csv(out / "test_per_sample.csv")

    config = dict(k=best_k, hidden=list(SIZES[best_size]), augment=bool(use_aug), post=post)
    A.save_pipeline(final, config, out / "model")
    df = pd.DataFrame(rows)
    df.round(5).to_csv(out / "results.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(dict(
        config=config, test=m_te, inference_ms_single=ms_single, inference_ms_per_sample_batch=ms_batch,
        split=dict(train=len(itr), validation=len(iva), test=len(test["density"])), runtime_s=time.time() - t_start), indent=2))

    # figures
    plot_pca_and_models(df, np.cumsum(pcas[max(KS)].explained_variance_ratio_), fig_dir)
    val = df[df.split == "val"]
    sz = val[val.experiment == "MLP size"].groupby("detail", sort=False)[["dice_free", "train_loss", "val_loss"]].agg(["mean", "std"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].errorbar(range(len(sz)), sz[("dice_free", "mean")], sz[("dice_free", "std")], fmt="o-", capsize=3)
    ax[0].set(xticks=range(len(sz)), xticklabels=sz.index, ylabel="Dice on free voxels (validation)", title=f"MLP size (k={best_k})")
    for name in ("train_loss", "val_loss"):
        ax[1].plot(sz[(name, "mean")].values, "o-", label=name.split("_")[0])
    ax[1].set(xticks=range(len(sz)), xticklabels=sz.index, ylabel="MSE on PCA coefficients", yscale="log", title="Capacity and generalisation")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "3_mlp_sizes.png")

    h = final.history
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(h[:, 0], label="training")
    ax.plot(h[:, 1], label="validation")
    ax.axvline(h[:, 1].argmin(), ls=":", c="grey", label="best epoch (early stopping)")
    ax.set(xlabel="epoch", ylabel="MSE on PCA coefficients", yscale="log", title="Training curves of the selected MLP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "4_training_curves.png")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    ax[0].hist(ps.dice_free, bins=20)
    ax[0].set(xlabel="Dice on free voxels", ylabel="test samples", title="Per-sample test accuracy")
    ax[1].scatter(ps.ref_voxels, ps.pred_voxels, s=10)
    lim = [ps.ref_voxels.min() - 20, ps.ref_voxels.max() + 20]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set(xlabel="material voxels, reference", ylabel="material voxels, prediction", title="Material volume")
    ax[2].scatter(ps.direction_deg, ps.dice_free, s=10)
    ax[2].set(xlabel="load direction (degrees)", ylabel="Dice on free voxels", title="Accuracy by load direction")
    fig.tight_layout()
    fig.savefig(fig_dir / "5_test_results.png")
    print(f"done in {time.time() - t_start:.0f} s; test: {m_te}")


if __name__ == "__main__":
    main()
