# AeroTopoML

Predicting optimised 3D material layouts of a lightweight bracket-like disc from its load conditions, with **PCA + MLP** on the
SELTO `disc_simple` dataset, plus a **Studio** with 18 examples in 6 categories: 14 real-world-inspired parts (nut, bolt, spanner,
drone frame, wing rib, car chassis, brake pedal, crane hook, lattice tower, ...) optimised by a built-in 3D optimiser, and 4 real SELTO
problems shown next to the dataset's own solution. CPU only, no GPU needed. A research prototype, not a certified design tool.

## AeroTopoML Studio: pick a part, press Optimize

`streamlit run src/app.py` opens the Studio. Pick a **category**, choose a part from the list, press **Optimize**, follow the pipeline
as it runs, then compare the **INITIAL** design space with the **FINAL** optimised part in 3D: drag to rotate, replay the optimisation
with the slider, download an STL. The last category, **Real dataset examples**, adds a third panel: the dataset's own solution.

| Category | Part | Loads and supports | Keeps | Stiffness kept | Time |
|---|---|---|---|---|---|
| Fasteners & tools | Hex nut | 20 kN thread load, washer face fixed | 50% | 79% | 6 s |
| | Hex bolt | 15 kN tension, 5 kN sideways and a tightening torque on the head, head underside fixed | 60% | 83% | 11 s |
| | Ring spanner | 400 N on the end of the handle, faces of the hex opening fixed | 40% | 52% | 4 s |
| Aerospace & drones | Drone frame | 12 N thrust and rotor torque on four motor mounts, central deck fixed | 50% | 45% | 6 s |
| | Aircraft wing rib | 9 kN air pressure on the skin between the spars, both spars fixed | 60% | 73% | 5 s |
| Automotive | Car chassis | engine, cabin and cargo (15 kN) on top, four suspension mounts fixed | 20% | 29% | 10 s |
| | Suspension control arm | 4 kN cornering force and 1.5 kN bump at the ball joint, two bushings fixed | 30% | 48% | 6 s |
| | Brake pedal | 600 N foot force on the pad, pivot fixed, push rod holds along one axis | 50% | 65% | 5 s |
| | Connecting rod | 15 kN pull on the small end with a sideways whip, bore of the big end fixed | 45% | 66% | 4 s |
| | Car wheel (SELTO) | 3 kN on the rim, hub fixed; any of the 200 test load cases | reference's | varies | ML: 2 ms, physics: 6 s |
| Machines & lifting | Crane hook | 20 kN resting in the bowl, top of the shank fixed | 50% | 90% | 5 s |
| | Wall bracket | 1.5 kN hanging load on a pin, four wall bolts fixed | 30% | 44% | 13 s |
| Structures & furniture | Footbridge | 5 kPa crowd load on the deck, pinned and rolling abutments | 40% | 44% | 13 s |
| | Lattice tower | 30 kN sideways and 10 kN weight at the top, foundation fixed | 35% | 44% | 8 s |
| | Stool | 1.5 kN on the seat, four feet fixed | 30% | 29% | 17 s |
| Real dataset examples (SELTO) | Disc, two loads | 3 kN in two loads on the rim, hub fixed; 200 load cases | reference's | varies | 5 s |
| | Sphere, one load | 3 kN in one load, the dataset's supports; 36 load cases, 39 x 39 x 21 voxels | reference's | varies | 30 s |
| | Sphere, two loads | the same with two loads; 36 load cases | reference's | varies | 40 s |

"Keeps" is the share of the mass kept (adjustable in the app), "Stiffness kept" the stiffness of the final part relative to the full
design space under the same loads. Stiffness per kg is above 1 for 12 of the 14 built parts; for the drone frame (0.90) and the stool
(0.98) the part loses stiffness about as fast as mass. Times are for 50 iterations at normal detail on this machine (16 threads).

**The pipeline** (shown live): voxelise the design space, apply supports and loads, build a finite-element model (8-node hexahedra),
iterate SIMP topology optimisation (density filter, optimality-criteria update, live flexibility chart), extract exactly the chosen mass
and drop pieces that carry nothing, verify the part with the same finite-element model, (for the dataset examples) compare with the
dataset's own solution, render. The wheel can instead run the trained **PCA + MLP** (features, MLP, inverse PCA, threshold,
connectivity check, FE verification), which answers in milliseconds. There is no trained model for the other parts, so they use the
physics optimiser (`src/topopt.py`, `src/objects.py`); the multi-threaded
`pypardiso` solver makes an iteration about 9 times faster than SciPy's SuperLU (0.23 s against 2.0 s on the 8,600-voxel bracket,
10-core CPU), and the code falls back to SuperLU without it.

**Does the optimiser agree with the dataset?** On 14 test samples (every 15th) it was given the same problem and the same amount
of material as the SELTO reference (made with OptiStruct SIMP):

| | overlap with the reference (Dice, free voxels) | stiffness relative to the reference (same FE model) |
|---|---|---|
| this optimiser (6 s each) | mean 0.84, from 0.50 to 0.98 | median 1.04, from 1.00 to 1.13 |
| trained PCA + MLP (0.5 ms) | mean 0.91 | median 1.03 |

The same check on the three extra subsets (a spot check: 4 samples each, evenly spread over the test set, 50 iterations):

| subset | overlap with the reference (Dice, free voxels) | stiffness relative to the reference |
|---|---|---|
| `disc_complex`: two loads, 200 samples | 0.60 to 0.83, mean 0.77 | 1.03 to 1.10 |
| `sphere_simple`: one load, 36 samples | 0.81 to 0.93, mean 0.89 | 1.01 |
| `sphere_complex`: two loads, 36 samples | 0.79 to 0.94, mean 0.89 | 1.00 to 1.02 |

So its parts are as stiff as the reference solutions (on the two-load discs a little stiffer), though not always the same shape:
topology optimisation has no unique optimum, and the dataset's solver may add constraints (stress) that this one does not. The overlap
is lowest on the two-load discs (0.60 on the worst of the four); I did not look into why. `tests/test_topopt.py` also checks the
element stiffness, the sensitivities against finite differences, the parts and dataset examples, and one such sample.

**More data.** I looked for ready-made optimised structures of nuts, bolts, drones and cars and found none that is open and has loads
and optimised results; SELTO's other subsets (discs and spheres) are the closest and are in the Studio:

| Dataset | What it has | Why it is not used here |
|---|---|---|
| [SELTO](https://zenodo.org/record/7781392) other subsets (`disc_complex`, `sphere_simple`, `sphere_complex`) | 3D problems with their optimised structures, same format | **used**: the three test sets (6.0 MB) are the "Real dataset examples"; their training sets (113 MB) are not downloaded |
| [TopoDroneX](https://huggingface.co/datasets/IAI-CAU/TopoDroneX) | 1,961 topology-optimised drone frames (meshes, thrust, stress) | gated: it asks you to accept terms and share contact details, so only you can request it |
| [MCB](https://engineering.purdue.edu/cdesign/wp/a-large-scale-annotated-mechanical-components-benchmark-for-classification-and-retrieval-tasks-with-deep-neural-networks/) (58,696 parts incl. nuts, bolts), [ABC](https://cims.nyu.edu/gcl/datasets.html) (1M CAD models), [Fusion 360 Gallery](https://arxiv.org/abs/2010.02392) (8,625 designs), Thingi10K | CAD and mesh geometry | geometry only, no loads and no optimised result; mostly large downloads |
| [DrivAerNet++](https://arxiv.org/abs/2406.09624) | 8,000 car bodies with CFD | aerodynamics, not structure; about 39 TB, non-commercial licence |

So the 14 built parts are computed by the solver, the way SELTO itself was made. To train an ML model for one of them, generate load
cases with `topopt.optimize` and fit PCA + MLP as for the wheel.

**Limits of the Studio.** The parts are simplified scale models: the loads are plausible, not design loads, the masses are those of
solid material filling the voxels (the "bridge" is solid concrete), and no stress, buckling or manufacturing limit is checked. Voxel
elements are too stiff in bending and the parts are only 20 to 60 voxels across, so read the stiffness numbers as a comparison between
the initial and the final part, not as absolute values (see "Mechanical check"). Surfaces in the 3D view are smoothed from the voxels.

## See the result in 3D (before and after)

| What | Where |
|---|---|
| Interactive page, 20 test samples, 4 linked 3D views (drag to rotate, scroll to zoom) | `outputs/before_after_3d.html`, just open it in a browser |
| Interactive app: the Studio above, and a second page for the 200 test samples with live prediction, STL download, what-if load direction | `streamlit run src/app.py` |
| Static renders (PyVista), best / 75th percentile / median / 25th percentile / worst sample | `outputs/3d/gallery.png`, `outputs/3d/3d_test_XXX.png` |
| Native PyVista window for one sample | `python src/viz3d.py --show 17` |
| Candidate STL files, reference and prediction, in millimetres | `outputs/3d/stl/` |

Each view shows: design problem (loads and supports) | **BEFORE** = reference from conventional topology optimisation |
**AFTER** = ML prediction | difference (grey = in both, blue = reference only, orange = prediction only).

## Results

SELTO `disc_simple`: 39 x 39 x 4 voxels (97.5 x 97.5 x 10 mm plate), one load per sample. 1,509 training samples were split
into 1,208 train / 301 internal validation; the 200 supplied test samples were used once, at the end.

"Free voxels" are the voxels the optimiser decides. The fixed regions (hub, supports, load patch, outside of the disc) are known in
advance, so predicting *nothing* already scores Dice 0.585 on all voxels. The free-voxel score is the honest one.

**Test set (200 samples)**

| Method | Dice, all voxels | Dice, free voxels | IoU, free voxels | Volume error | Load joined to supports |
|---|---|---|---|---|---|
| Fixed regions only (no learning) | 0.585 | 0.000 | 0.000 | 59% | 0% |
| Linear regression + PCA(128) | 0.782 | 0.585 | 0.428 | 21% | 20% |
| 1-nearest-neighbour lookup (reference only) | 0.889 | 0.810 | 0.696 | 5.6% | 98% |
| **PCA(128) + MLP 3x256 (final)** | **0.953** | **0.921** | **0.855** | 4.8% | 100% |
| PCA(128) reconstruction of the true structures (ceiling for any PCA(128) model) | 0.969 | 0.947 | 0.900 | 1.6% | 100% |

Per sample, free-voxel Dice has a median of 0.928; 78% of the test samples score at least 0.90, the worst scores 0.768.
Every predicted structure connects the load to the supports. Before the mandatory regions are enforced, under 0.1% of the fixed voxels are wrong.
Prediction takes about 0.5 ms per sample on the CPU (loads in, binary structure out).

**Experiments from the proposal** (internal validation, Dice on free voxels, mean of 3 seeds (2 for augmentation), same split everywhere)

| Experiment | Result |
|---|---|
| PCA reconstruction ceiling, k = 32 / 64 / 128 / 256 | 0.858 / 0.919 / 0.947 / 0.963 |
| Linear regression + PCA, same k | 0.561 / 0.564 / 0.564 / 0.563 |
| MLP 2x128 + PCA, same k | 0.841 / 0.888 / 0.900 / 0.903 |
| MLP size at k = 128: 2x32, 2x64, 2x128, 2x256, 3x256 | 0.823, 0.879, 0.900, 0.907, 0.912 |
| Input features: plain cos/sin (8 numbers) vs with angle harmonics (20 numbers) | 0.885 vs 0.900 |
| Symmetry augmentation (8x training data), 3x256 | 0.917 vs 0.912 without |
| Threshold | best at 0.45 to 0.50; reconnecting or cleaning the structures changed nothing (already connected) |

So the nonlinear MLP matters a lot (0.56 to 0.90): the structure depends on the load in a way a linear map cannot follow.
Selected: k = 128 (the smallest k within 0.005 of the best), MLP 3x256, 8x augmentation, threshold 0.45.
Everything is in `outputs/results.csv`, `outputs/metrics.json`, `outputs/test_per_sample.csv`, `outputs/figures/`.

## Mechanical check (voxel finite elements)

Overlap is not performance, so `src/fem.py` solves the same load case on the reference and on the predicted structure (the config's
post-processing) of every test sample, and compares the **compliance** (load times displacement, joule), the **largest displacement**
and the **stiffness ratio** C_reference / C_prediction (same load, so above 1 means the prediction is stiffer).
Model: linear elasticity, one trilinear 8-node hexahedron per 2.5 mm voxel (E = 70 GPa, nu = 0.3), sparse direct solve, void voxels keep
1e-9 of the solid stiffness, a support voxel fixes all 8 of its corner nodes. The 400 solves take about 6 minutes.
Results: `outputs/fem_per_sample.csv`, `outputs/fem_summary.json`. `tests/test_fem.py` checks the solver: uniform tension is reproduced
exactly, void padding changes nothing, a cantilever converges to beam theory, the unit conversion, more material never lowers stiffness.

**Unit assumption.** `force_val` is a *body force density in N/m^3* (SELTO Zenodo record: "All forces are body forces given in
[N/m^3]"; paper, section 3), not newtons per voxel. A loaded voxel therefore carries F * h^3 newtons (h = 2.5 mm), an eighth on each of
its corner nodes. This gives total loads of 1500, 1667, 1750 or 2000 N per sample, round numbers; read as newtons per voxel they would be
about 1e11 N. The papers do not say how the reference solver (OptiStruct) maps voxels to nodes, so the support rule above is my assumption.

**What it shows** (200 test samples, prediction / reference)

| | stiffness | largest displacement |
|---|---|---|
| median | 1.031 | 0.996 |
| 5th to 95th percentile | 0.917 to 1.175 | 0.787 to 1.308 |
| min to max | 0.566 to 1.587 | 0.684 to 2.465 |
| share within +-10% / +-20% | 75.5% / 95% | 57% / 77.5% |

- **Stiffness is mostly kept but biased high**: the prediction is stiffer in 72.5% of the samples, because it holds more material. Its median
  volume is 3.3% above the reference (76% of the samples are heavier; the threshold 0.45 chosen for Dice adds material, with 0.5 the mean
  volume ratio is 1.003), the stiffness ratio follows the volume ratio (Spearman 0.91), and per voxel of material the prediction is as
  stiff as the reference (median 1.003, 5th to 95th percentile 0.95 to 1.06). It is a slightly heavier design, not a better one.
- **The largest displacement scatters more**: 43% of the samples are more than 10% off, since it is a local value: the peak sits at the load
  patch (within 2 voxels of a loaded voxel in the 20 reference structures I looked at).
- **Overlap helps but does not guarantee equal stiffness** (Spearman -0.68 between Dice and the size of the stiffness error):

| Dice, free voxels | samples | median stiffness ratio | within +-10% | range |
|---|---|---|---|---|
| >= 0.95 | 38 | 1.006 | 97% | 0.95 to 1.12 |
| 0.90 to 0.95 | 119 | 1.040 | 81% | 0.87 to 1.18 |
| < 0.90 | 43 | 1.083 | 42% | 0.57 to 1.59 |

  Of the 157 samples with Dice >= 0.90, 24 differ in stiffness by more than 10% and none by more than 20% (listed in `fem_summary.json`).
  23 of them are stiffer by 10 to 18% and carry 5 to 11% more material; sample 160 is 13% softer with 9% less. The worst prediction is
  sample 128: Dice 0.895 and 3% more material, yet 43% softer (ratio 0.57) with 2.5 times the peak displacement. The cause is local: adding
  back the 8 reference voxels it lacks within 3 voxels of the load patch lifts its ratio to 1.03 (samples 94 and 21, the next softest,
  behave the same: 12 and 15 voxels give 0.98 and 1.00). A few missing voxels where the load enters can cost more stiffness than a
  larger error elsewhere. The other extreme is sample 197: 59% stiffer with 12% more material (Dice 0.825).

**Limits.**
- Voxel FE with trilinear elements is too stiff in bending, and the members here are 1 to 3 voxels wide: a cantilever test reaches only 87 /
  95 / 98% of the beam-theory deflection with 2 / 4 / 8 elements across the thickness (64% with 1). Absolute compliance and displacement are
  therefore not accurate; only the comparison of two structures under the same model means something. Voxels that touch along an edge or a
  corner share nodes and carry load in this model.
- No stress: yield (450 MPa), stress concentrations and buckling were not checked, and a stiffer structure is not a safer one.
- Void voxels keep 1e-9 of the solid stiffness (a one-off check with 1e-6 changed the ratios by less than 1e-5), so material that is
  connected to neither the supports nor the load carries nothing. One linear load case per sample, small displacements.
- The references are binarised results of another solver (OptiStruct SIMP) evaluated in this model, not the optimiser's own numbers. A
  one-off check with a stricter support rule (a node is fixed only if every voxel around it is a support) changed the absolute compliance by
  8 to 28% but the stiffness ratio by at most 2.4% (12 samples). These one-off checks, the peak position and the restored voxels above are
  scratch scripts, not part of the repo.
- Not done: the same comparison at threshold 0.5 (equal volume), which would separate the material effect from the layout effect.

## How it works

1. `prepare.py` parses the CSVs. Supports and material constants are identical in all samples (asserted), so they are not features.
2. Load conditions become 20 numbers: load position, and cos / sin of three harmonics of the load direction, the load position on the
   rim and the direction relative to the radial direction. Only in-plane loads occur (Fz = 0 everywhere).
3. PCA is fitted on the training structures only. An MLP (ReLU, Adam, MSE, early stopping, learning rate halved on plateaus) maps the
   20 numbers to the PCA coefficients.
4. Prediction = inverse PCA, threshold, then the mandatory solid and void regions are enforced. Connectivity is checked
   (`reconnect` and `clean` options exist for structures that fall apart).

## Differences from the proposal

- Extra metrics on free voxels, and two extra references (fixed regions only, nearest neighbour), so the Dice numbers can be judged.
- Angle harmonics in the features, a learning-rate schedule and a wider MLP than the proposed 2x128 (2x128 is in the comparison).
- Symmetry augmentation: the disc, hub and supports are symmetric, so rotated and mirrored problems are valid extra training data.
- Not done: PCA compression of the *inputs* (the proposal only says it can be investigated; 20 numbers already describe the load).
  The optional mechanical (FEM) comparison of stiffness and displacement was added afterwards, see "Mechanical check".

## Limits

- Overlap is not mechanical performance. Stiffness and displacement of the predictions were checked with a coarse voxel FE model
  ("Mechanical check"); stress was not.
- The test samples come from the same family as the training samples (about 48 load positions on the rim, directions mostly in 10
  degree steps), so the scores measure interpolation inside that family, not new geometries or load types.
- Predicted material volumes are less spread out than the reference (correlation 0.72): the model regresses toward the mean.
- STL files are raw voxel surfaces (2.5 mm voxels). Some have non-manifold edges where voxels touch along an edge; they need
  smoothing, wall-thickness and manufacturability checks before any printing.
- The references are one optimiser's answer; topology optimisation has no unique optimum. The run time of conventional optimisation
  was not measured here, so no speed-up factor is claimed.

## Set up on another machine

Built and tested with Python 3.13 on Windows 11, CPU only. The repository holds the code, the processed data (`data/processed`), the
trained model and all results (`outputs/`); only the downloaded archives and their unpacked CSVs (about 425 MB) are left out, and
`src/fetch_data.py` rebuilds them. So after installing the requirements the app starts at once:

```
git clone https://github.com/Balasanthoshraj/AeroTopoML.git
cd AeroTopoML
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt       # macOS / Linux: .venv/bin/python
.venv\Scripts\streamlit run src/app.py
```

The environment takes about 2 GB once installed (PyTorch, Intel MKL for the optional fast solver, PyVista with VTK). On Linux the
default PyTorch wheel also pulls the CUDA libraries; for a CPU-only install run
`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` first (not tested here, I only ran this on Windows).

## Run it yourself

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -I src/fetch_data.py data                 # 23.9 MB from Zenodo, md5-checked (17.9 MB of it is disc_simple)
.venv\Scripts\python -I src/prepare.py data/raw data/processed
.venv\Scripts\python tests/test_core.py
.venv\Scripts\python tests/test_topopt.py
.venv\Scripts\python tests/test_fem.py
.venv\Scripts\python src/run_experiments.py                    # about 25 minutes on a laptop CPU
.venv\Scripts\python src/fem.py                                # voxel FE check of the 200 test samples, about 6 minutes
.venv\Scripts\python src/viz3d.py
.venv\Scripts\streamlit run src/app.py
```

`-I` (isolated mode) is used for the two scripts that read the downloaded files, which are treated as untrusted data.

## Layout

```
src/fetch_data.py       download, verify, extract the dataset
src/prepare.py          CSV -> compact arrays (data/processed)
src/aerotopoml.py       features, PCA + MLP, post-processing, metrics
src/run_experiments.py  all comparisons, final test evaluation, figures, saved model
src/viz3d.py            3D views: PNG, interactive HTML (before / after, optimisation replay), smooth and voxel STL, native window
src/topopt.py           3D SIMP topology optimiser with hexahedral finite elements (PARDISO or SuperLU)
src/objects.py          the parts library (14 parts in 6 categories) and the SELTO subsets as problems with their reference solution
src/fem.py              voxel finite-element solver, stiffness / displacement of reference vs prediction
src/app.py              Streamlit: the Studio (pick a part, Optimize) and the ML study page
tests/test_core.py      self-checks (symmetry augmentation, meshing, metrics, data codes)
tests/test_topopt.py    self-checks of the optimiser (element stiffness, sensitivities, a run, the parts, agreement with the dataset)
tests/test_fem.py       self-checks of the FE solver (exact tension patch test, void padding, beam convergence, units)
outputs/                results, figures, model, 3D files, run_log.txt, fem_per_sample.csv, fem_summary.json
```

Data: SELTO dataset, Zenodo [10.5281/zenodo.7781392](https://zenodo.org/record/7781392) (CC BY 4.0); Dittmer et al.,
*SELTO: Sample-Efficient Learned Topology Optimization*, [arXiv:2209.05098](https://arxiv.org/abs/2209.05098).
