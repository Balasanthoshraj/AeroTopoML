"""AeroTopoML Studio: choose a part, press Optimize, follow the pipeline, compare the initial and the optimised part in 3D.

  streamlit run src/app.py
"""
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import aerotopoml as A
import objects as O
import topopt as T
import viz3d

st.set_page_config(page_title="AeroTopoML Studio", layout="wide")
NOTE = "Research prototype with simplified scale models: the loads are plausible, not certified design loads."


@st.cache_resource
def ml():
    pipe, cfg = A.load_pipeline()
    return A.load("test"), pipe, cfg


@st.cache_resource
def dataset(name):
    """Arrays of a processed SELTO subset (data/processed/<name>.npz)."""
    with np.load(A.DATA / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def dice(a, b, free):
    """Overlap of two structures on the voxels the optimiser decides."""
    return float(2 * (a & b & free).sum() / max((a & free).sum() + (b & free).sum(), 1))


def stl_bytes(structure, voxel_mm, smooth=True):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "part.stl"
        if smooth:
            viz3d.export_stl_smooth(path, structure, voxel_mm)
        else:  # (an `x if c else y` statement would be echoed by Streamlit's magic as "None")
            viz3d.export_stl(path, structure, voxel_mm / 1000)
        return path.read_bytes()


class Log:
    """Shows each pipeline step as it happens and keeps the lines, so they can be shown again later."""

    def __init__(self):
        self.lines = []

    def __call__(self, text):
        self.lines.append(text)
        st.markdown(text)


def counts(p):
    act = p.design != T.VOID
    return dict(active=int(act.sum()), solid=int((p.design == T.SOLID).sum()), free=int((p.design == T.FREE).sum()),
                supported=int(p.fixed.any(-1)[act].sum()), loaded=int((np.abs(p.force).sum(-1) > 0).sum()),
                kN=float(np.linalg.norm(p.force, axis=-1).sum()) / 1000)  # all loads added up, whatever their directions


def pipeline_physics(p, spec, iterations, ref=None):
    """The classic pipeline: voxelise, constrain, build the FE model, iterate SIMP, extract, verify. With a dataset reference
    structure a last step compares with it. Returns (Result, log lines)."""
    log, c, t0 = Log(), counts(p), time.perf_counter()
    with st.status("Running the optimisation pipeline", expanded=True) as status:
        log(f"**1 · Load part**: {spec.label}, {p.design.shape[0]} × {p.design.shape[1]} × {p.design.shape[2]} voxels of {p.voxel_mm:.1f} mm, {spec.material_name}")
        log(f"**2 · Voxelise the design space**: {c['active']:,} voxels, {c['free']:,} free to optimise and {c['solid']:,} fixed")
        log(f"**3 · Apply supports and loads**: {c['supported']:,} supported voxels, {c['loaded']:,} loaded voxels, {c['kN']:.1f} kN in total")
        slot4 = st.empty()  # step 4 is reported while the optimiser runs, but belongs above its progress bar
        bar, chart, first, rows, seen = st.progress(0.0, text="4 · Build the finite-element model"), st.empty(), [None], [], {}

        def on(stage, **k):
            if stage == "model":
                text = f"**4 · Build the finite-element model**: {k['elements']:,} hexahedral elements, {k['dofs']:,} unknowns ({k['seconds']:.1f} s)"
                slot4.markdown(text)
                log.lines.append(text)
                bar.progress(0.02, text=f"5 · Optimise (SIMP), iteration 0 of {iterations}")
            elif stage == "iteration":
                first[0] = first[0] or k["compliance"]
                rows.append(k["compliance"] / first[0])
                bar.progress(min(k["it"] / k["total"], 1.0), text=f"5 · Optimise (SIMP), iteration {k['it']} of {k['total']}: "
                                                                 f"mass {k['mass']:.0%} of the start, flexibility {rows[-1]:.2f} × the first iteration")
                if k["it"] % 3 == 0 or k["it"] == k["total"]:
                    chart.line_chart(pd.DataFrame({"flexibility (lower is stiffer)": rows}), height=130)
            elif stage == "extract":
                seen.update(k)

        res = T.optimize(p, iterations=iterations, callback=on)
        m = res.metrics
        log(f"**5 · Optimise (SIMP)**: {m['iterations']} iterations in {m['seconds']:.1f} s, keeping the stiffest {1 - m['mass_saved']:.0%} of the mass")
        log(f"**6 · Extract the part**: exactly that mass, and {seen.get('removed', 0)} floating voxel(s) that carried no load removed")
        log(f"**7 · Verify with the finite-element model**: {m['stiffness_retained']:.0%} of the stiffness kept at {1 - m['mass_saved']:.0%} of the mass")
        if ref is not None:
            m.update(ref_stiffness=T.evaluate(p, ref)["stiffness_retained"], dice=dice(res.structure, ref, p.design == T.FREE))
            log(f"**8 · Compare with the dataset's own solution**: it keeps {m['ref_stiffness']:.0%} of the stiffness at the same mass, "
                f"this one {m['stiffness_retained']:.0%}; the two overlap with Dice {m['dice']:.2f} on the free voxels")
        bar.progress(1.0, text="Optimisation finished")
        status.update(label=f"Pipeline finished in {time.perf_counter() - t0:.1f} s", state="complete", expanded=False)
    return res, log.lines


def pipeline_ml(p, d, i, pipe, cfg):
    """The ML pipeline of this project on a SELTO sample: features, MLP, inverse PCA, threshold, check, verify."""
    log, one, post = Log(), A.take(d, [i]), cfg["post"]
    f = one["force"][0]
    with st.status("Running the ML pipeline", expanded=True) as status:
        log(f"**1 · Load the load case**: test sample #{i}, {counts(p)['kN'] * 1000:.0f} N on the rim at "
            f"{np.degrees(np.arctan2(one['centroid'][0][1] - A.AXIS, one['centroid'][0][0] - A.AXIS)):.0f}°, pointing {np.degrees(np.arctan2(f[1], f[0])):.0f}°")
        log(f"**2 · Describe the load with {one['x'].shape[1]} numbers**: its position and the harmonics of its angles")
        t = time.perf_counter()
        coef = pipe.coef(one["x"])
        t_mlp = time.perf_counter() - t
        log(f"**3 · The MLP predicts {coef.shape[1]} PCA coefficients** ({1000 * t_mlp:.2f} ms)")
        t = time.perf_counter()
        dens = pipe.pca.inverse_transform(coef).reshape(-1, *A.SHAPE)
        log(f"**4 · Inverse PCA rebuilds the density of all {dens[0].size:,} voxels** ({1000 * (time.perf_counter() - t):.2f} ms)")
        pred = A.structures(dens, one, **post)[0]
        log(f"**5 · Threshold at {post['tau']:.2f} and enforce the fixed regions**: {int(pred.sum())} voxels of material")
        pieces, joined = A.connected(pred, one["load"][0], one["support"][0])
        log(f"**6 · Check the structure**: {pieces} piece(s), the load is {'joined to' if joined else 'NOT joined to'} the supports")
        m = T.evaluate(p, pred)
        m.update(iterations=0, seconds=t_mlp, dice=dice(pred, d["density"][i], p.design == T.FREE))
        log(f"**7 · Verify with the finite-element model**: {m['stiffness_retained']:.0%} of the stiffness kept at {1 - m['mass_saved']:.0%} of the mass; "
            f"agrees with the reference optimiser (Dice {m['dice']:.2f} on free voxels)")
        status.update(label=f"ML prediction finished, the network itself took {1000 * t_mlp:.2f} ms", state="complete", expanded=False)
    return T.Result(pred, dens[0] * pred, [], [], m), log.lines


def show(p, res, key, smooth, reference=None):
    """Metrics, the 3D comparison and the downloads of a finished run. A dataset reference structure is drawn next to the result."""
    m = res.metrics
    c = st.columns(5)
    c[0].metric("Mass after", viz3d.fmt_mass(m["mass_final_g"]), f"-{m['mass_saved']:.0%} from {viz3d.fmt_mass(m['mass_initial_g'])}", delta_color="inverse")
    c[1].metric("Stiffness kept", f"{m['stiffness_retained']:.0%}", help="Stiffness of the final part divided by the stiffness of the full design space, same loads and supports.")
    c[2].metric("Stiffness per kg", f"× {m['stiffness_per_mass_gain']:.2f}", help="Above 1: the part is stiffer for its weight than the full block. Below 1: material was removed faster than stiffness was lost.")
    if m["iterations"]:
        c[3].metric("Optimisation time", f"{m['seconds']:.1f} s")
    else:
        c[3].metric("Network time", f"{1000 * m['seconds']:.2f} ms")
    if "dice" in m:
        c[4].metric("Dice vs dataset", f"{m['dice']:.2f}", help="Overlap with the dataset's own solution, on the voxels the optimiser decides (1 = identical).")
    else:
        c[4].metric("Iterations", m["iterations"])
    items = viz3d.studio_items(p, res, smooth, reference)
    html = viz3d.html_page(items, picker="slider" if len(items) > 1 else False, embed=True, legend=viz3d.LEGEND_STUDIO if reference is None else viz3d.LEGEND_STUDIO_REF)
    st.iframe(html, height="content")
    a, b, _ = st.columns([1, 1, 2])
    a.download_button("Download smooth STL", stl_bytes(res.structure, p.voxel_mm), f"{key}_optimised.stl", help="Millimetres. Candidate geometry only: check wall thickness and strength before manufacturing.")
    b.download_button("Download voxel STL", stl_bytes(res.structure, p.voxel_mm, smooth=False), f"{key}_optimised_voxels.stl")


def studio():
    st.title("AeroTopoML Studio")
    st.caption("Choose a part, press Optimize, follow the pipeline, then compare the initial design space with the optimised part in 3D. " + NOTE)
    runs = st.session_state.setdefault("runs", {})
    left, right = st.columns([1, 3.4], gap="large")
    with left:
        st.markdown("##### 1 · Choose a part")
        group = st.selectbox("Category", list(dict.fromkeys(s.group for s in O.CATALOG.values())), label_visibility="collapsed")
        key = st.radio("Part", [k for k, s in O.CATALOG.items() if s.group == group], format_func=lambda k: f"{O.CATALOG[k].icon}  {O.CATALOG[k].label}", label_visibility="collapsed")
        spec = O.CATALOG[key]
        st.markdown("##### 2 · Settings")
        ref = None
        if spec.dataset:  # a problem from the SELTO dataset, which also comes with the dataset's own solution
            if key == "wheel":
                d, pipe, cfg = ml()
            elif (A.DATA / f"{spec.dataset}.npz").exists():
                d = dataset(spec.dataset)
            else:
                st.error("The data of this example is not prepared: run src/fetch_data.py and src/prepare.py (see the README).")
                st.stop()
            sample = st.slider("Load case (dataset sample)", 0, len(d["density"]) - 1, 82 if key == "wheel" else 0, key=f"sample_{key}")
            engine = st.radio("Engine", ["ML model (PCA + MLP)", "Physics optimiser (SIMP)"]) if key == "wheel" else "Physics optimiser (SIMP)"
            p, setting, use_ml, ref = O.dataset_problem(d, sample, spec.label), (sample, engine), engine.startswith("ML"), d["density"][sample].astype(bool)
        else:
            keep = st.slider("Material to keep (%)", 10, 90, round(spec.keep * 100), key=f"keep_{key}") / 100
            detail = st.select_slider("Detail", list(O.DETAIL), "Normal", key=f"detail_{key}", help="Finer voxels give smoother parts but take longer.")
            p, setting, use_ml = O.build(key, detail, keep), (keep, detail), False
        iterations = 50 if use_ml else st.slider("Iterations", 20, 100, 50, 5)
        n = counts(p)["active"]
        slower = 1 if T.PyPardisoSolver else 9  # SuperLU is about 9 times slower than PARDISO
        st.caption("ML answers instantly." if use_ml else f"{n:,} voxels: about {slower * iterations * 0.28 * (n / 9400) ** 1.8:.0f} s on this machine.")
        smooth = st.checkbox("Smooth surface", True, help="Rounds off the voxel steps in the 3D view of the final part. The part itself is the same.")
        go = st.button("⚙️  Optimize", type="primary", use_container_width=True)
    sig = (key, setting, iterations)
    with right:
        st.subheader(f"{spec.icon}  {spec.label}")
        st.write(spec.blurb)
        st.markdown("\n".join(f"- {f}" for f in spec.facts) + f"\n- Material: {spec.material_name}, voxels of {p.voxel_mm:.1f} mm")
        if go:
            try:
                T.validate(p)
                res, lines = pipeline_ml(p, d, sample, pipe, cfg) if use_ml else pipeline_physics(p, spec, iterations, ref)
                runs[sig] = dict(p=p, res=res, lines=lines)
                for old in list(runs)[:-3]:  # keep the last three runs only
                    del runs[old]
            except ValueError as e:
                st.error(f"This setting cannot be optimised: {e}")
        if sig in runs:
            run = runs[sig]
            with st.expander("Pipeline details"):
                st.markdown("\n\n".join(run["lines"]))
            show(run["p"], run["res"], key, smooth, ref)
        else:
            st.info("This is the INITIAL design space with its supports (green) and loads (red). Press **Optimize** to remove everything that is not needed.")
            html = viz3d.html_page([("Initial", "", viz3d.figure(viz3d.studio_view(p)))], picker=False, embed=True, legend=viz3d.LEGEND_STUDIO)
            st.iframe(html, height="content")


def study():
    d, pipe, cfg = ml()
    post, n = cfg["post"], len(d["density"])
    st.title("ML study: reference against prediction")
    st.caption("SELTO disc_simple test samples. Before = reference from conventional topology optimisation, after = PCA + MLP prediction. " + NOTE)
    left, right = st.columns([1, 3.4], gap="large")
    with left:
        i = int(st.number_input("Test sample", 0, n - 1, 0))
        theta0 = float(np.degrees(np.arctan2(d["force"][i][1], d["force"][i][0])))
        custom = st.checkbox("What-if: change the load direction (no reference exists)")
        theta = st.slider("Load direction (degrees)", -180, 180, int(round(theta0))) if custom else theta0
        tau = st.slider("Density threshold", 0.2, 0.8, float(post["tau"]), 0.05)
        repair = st.checkbox("Reconnect load and supports", post["repair"])
        clean = st.checkbox("Remove floating material", post["clean"])
    one = A.take(d, [i])
    force = one["force"][0].copy()
    if custom:
        force = np.linalg.norm(force) * np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta)), 0.0])
    t0 = time.perf_counter()
    dens = pipe.density(A.features(one["centroid"], force[None]))
    pred = A.structures(dens, one, tau=tau, repair=repair, clean=clean)
    ms = 1000 * (time.perf_counter() - t0)
    m = A.evaluate(dens, one, tau=tau, repair=repair, clean=clean)
    with right:
        cols = st.columns(5)
        cols[0].metric("Prediction time", f"{ms:.0f} ms")
        cols[1].metric("Material voxels (after)", int(pred.sum()))
        cols[2].metric("Load joined to supports", "yes" if m["joined"][0] else "no")
        if not custom:
            cols[3].metric("Dice, free voxels", f"{m['dice_free'][0]:.2f}")
            cols[4].metric("Material voxels (before)", int(one["density"].sum()))
        fig = viz3d.figure(viz3d.view(one, 0, pred[0], show_ref=not custom, force=force))
        st.iframe(viz3d.html_page([("", "", fig)], picker=False, embed=True), height="content")
        with tempfile.TemporaryDirectory() as tmp:
            stl = Path(tmp) / "predicted.stl"
            viz3d.export_stl(stl, pred[0], float(d["info"][3]))
            st.download_button("Download predicted structure (STL, millimetres)", stl.read_bytes(), f"test_{i:03d}_predicted.stl")
        st.caption("The STL is a raw voxel surface for inspection. Wall thickness, watertightness and strength still need to be checked before any manufacturing.")


page = st.radio("View", ["Optimise a part", "ML study: SELTO test samples"], horizontal=True, label_visibility="collapsed")
if page == "Optimise a part":
    studio()
else:
    study()
