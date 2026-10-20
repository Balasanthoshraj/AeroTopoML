"""3D before / after views of the reference topology and the PCA + MLP prediction.

  python src/viz3d.py              PNG renders (PyVista), an interactive HTML page (Plotly) and STL files for test samples
  python src/viz3d.py --show 17    interactive PyVista window for test sample 17 (rotate with the mouse)

Panels: design problem (loads, supports) | BEFORE: reference from conventional optimisation | AFTER: ML prediction | difference.
"""
import itertools
import json
import sys

import numpy as np
from scipy import ndimage

import aerotopoml as A
import topopt as T

# class ids in the label grids; class c is drawn with COLORS[c - 1]
CLASSES = ["fixed solid", "supports", "load patch", "reference material", "predicted material", "overlap", "design material"]
COLORS = ["#4a4f57", "#2e9e5b", "#d6403a", "#3b7dd8", "#f0802c", "#c9ccd1", "#8fa6bf"]
LEGEND_SELTO = list(zip(CLASSES[:6], COLORS[:6]))
LEGEND_STUDIO = [("fixed solid", COLORS[0]), ("supports", COLORS[1]), ("loads", COLORS[2]), ("design space", COLORS[6]), ("optimised material", COLORS[4])]
LEGEND_STUDIO_REF = LEGEND_STUDIO + [("dataset reference", COLORS[3])]
SPACE = "#aeb4bd"
CORNERS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


# ---------------------------------------------------------------- geometry
def voxel_mesh(lab):
    """Exposed faces of a labelled voxel grid (0 = empty): vertices (V, 3), outward-wound quads (Q, 4), class per quad."""
    pad, solid = np.pad(lab, 1), lab > 0
    verts, cls = [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3  # (a, b, c) cyclic, so corners in (b, c) order wind counter-clockwise seen from +a
        for s in (1, -1):
            p = np.argwhere(solid & (np.roll(pad, -s, axis=a)[1:-1, 1:-1, 1:-1] == 0))
            v = np.empty((len(p), 4, 3), np.int16)
            v[..., a] = (p[:, a] + (s > 0))[:, None]
            v[..., b] = p[:, b, None] + CORNERS[::s, 0]  # [::-1] flips the winding for faces pointing to -a
            v[..., c] = p[:, c, None] + CORNERS[::s, 1]
            verts.append(v.reshape(-1, 3))
            cls.append(lab[tuple(p.T)])
    verts, cls = np.concatenate(verts), np.concatenate(cls)
    if not len(verts):
        return np.zeros((0, 3), np.int16), np.zeros((0, 4), np.int64), cls
    uniq, inv = np.unique(verts, axis=0, return_inverse=True)
    return uniq, inv.reshape(-1, 4), cls


def export_stl(path, vox, voxel_size):
    """Write a boolean voxel grid as an STL surface in millimetres (candidate geometry, not checked for printing)."""
    import pyvista as pv
    verts, quads, _ = voxel_mesh(vox.astype(np.int8))
    pv.PolyData.from_regular_faces(verts * voxel_size * 1000.0, quads).triangulate().save(path)


# ---------------------------------------------------------------- what to draw
def view(d, i, pred, show_ref=True, force=None):
    """Label grids and titles of the panels for sample i of split d, given its predicted structure pred (39, 39, 4)."""
    ds, ref = d["design"][i], d["density"][i]
    free = ds == A.FREE
    base = np.zeros(A.SHAPE, np.int8)
    base[ds == A.SOLID] = 1
    base[(ds == A.SOLID) & d["dirichlet"].any(-1)] = 2
    base[d["load"][i]] = 3

    def material(mask, cls):
        g = base.copy()
        g[mask & free] = cls
        return g

    f = d["force"][i] if force is None else np.asarray(force, float)
    grids, titles = [base], ["Design problem\nloads and supports"]
    if show_ref:
        diff = base.copy()
        diff[free & ref & pred], diff[free & ref & ~pred], diff[free & ~ref & pred] = 6, 4, 5
        dice = A.overlap(pred[None], ref[None], free[None])[0][0]
        grids += [material(ref, 4), material(pred, 5), diff]
        titles += ["BEFORE\nreference (conventional optimisation)", "AFTER\nML prediction (PCA + MLP)",
                   f"Difference\nDice {dice:.2f} on free voxels"]
    else:
        grids.append(material(pred, 5))
        titles.append("AFTER\nML prediction (PCA + MLP)")
    u = f / np.linalg.norm(f)
    return dict(grids=grids, titles=titles, space=free, shape=A.SHAPE, arrows=[(d["centroid"][i] + 0.5 + 7.0 * u, u, 7.0)])  # (tip, direction, length)


# ---------------------------------------------------------------- interactive Plotly page
def default_eye(shape):
    """Camera direction: flat plates are seen from higher up, so their outline shows."""
    return (0.35, -1.5, 1.7) if max(shape[:2]) > 2.6 * shape[2] else (0.45, -2.1, 1.25)


def scene_zoom(shape, eye, aspect=1.9):
    """Scene units per voxel at which a box of this shape just fills a panel `aspect` times wider than tall, seen from eye.

    Plotly ties the scale of a 3D scene to the height of its panel: about 1.75 scene units span the panel height, aspect times that its width.
    """
    d = -np.array(eye, float)
    r = np.cross(d, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, d / np.linalg.norm(d))
    corners = np.array(list(itertools.product(*[(0, n) for n in shape])), float) - np.array(shape) / 2
    return 0.93 * min(1.746 / np.ptp(corners @ u), 1.746 * aspect / np.ptp(corners @ r))


def smooth_surface(structure, sigma=0.65):
    """Smooth triangle surface (pyvista PolyData) around a boolean voxel structure: marching cubes on a slightly blurred copy."""
    import pyvista as pv
    f = ndimage.gaussian_filter(np.pad(structure.astype(np.float32), 2), sigma)  # the padding lets the surface close
    grid = pv.ImageData(dimensions=f.shape, spacing=(1, 1, 1), origin=(-1.5, -1.5, -1.5))  # grid points sit on the voxel centres
    grid.point_data["f"] = f.ravel(order="F")
    return grid.contour([0.5], scalars="f").triangulate()


def vertex_colors(verts, classes, material=(240, 128, 44)):
    """Colour of each surface vertex: the region it lies in (fixed solid, support, load) or else the optimised material."""
    cls = classes[tuple(np.clip(np.floor(verts).astype(int), 0, np.array(classes.shape) - 1).T)]
    out = np.tile(np.array(material, np.uint8), (len(verts), 1))
    for c, rgb in {1: (74, 79, 87), 2: (46, 158, 91), 3: (214, 64, 58)}.items():
        out[cls == c] = rgb
    return out


def _smooth3d(go, structure, classes, scene, color=(240, 128, 44)):
    poly = smooth_surface(structure)
    verts, faces = np.asarray(poly.points, np.float32), poly.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], flatshading=False,
                     vertexcolor=vertex_colors(verts, classes, color), lighting=dict(ambient=0.55, diffuse=0.8, specular=0.25, roughness=0.5),
                     lightposition=dict(x=-200, y=-400, z=600), hoverinfo="skip", scene=scene)


def _mesh3d(go, lab, scene, color=None, opacity=1.0):
    verts, quads, cls = voxel_mesh(lab)
    tri = np.concatenate([quads[:, :3], quads[:, [0, 2, 3]]]).astype(np.uint16)
    kw = dict(color=color) if color else dict(
        intensity=np.tile(cls, 2).astype(np.int8), intensitymode="cell", cmin=0.5, cmax=len(COLORS) + 0.5, showscale=False,
        colorscale=[[t, c] for k, c in enumerate(COLORS) for t in (k / len(COLORS), (k + 1) / len(COLORS))])
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2], flatshading=True, opacity=opacity,
                     lighting=dict(ambient=0.62, diffuse=0.75, specular=0.08, roughness=0.9), lightposition=dict(x=-200, y=-400, z=600),
                     hoverinfo="skip", scene=scene, **kw)


def figure(v):
    """Plotly figure with one 3D scene per panel of view v (a dict from view() or studio_view()).

    Four panels go in a 2 x 2 grid (BEFORE and AFTER on top), fewer panels side by side.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    n, shape = len(v["grids"]), v["shape"]
    order = [1, 2, 0, 3] if n == 4 else list(range(n))  # panel shown in each grid cell, row by row
    rows, cols = (2, 2) if n == 4 else (1, n)
    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "scene"}] * cols] * rows, horizontal_spacing=0.0, vertical_spacing=0.07,
                        subplot_titles=[v["titles"][p].replace("\n", "<br>") for p in order])
    smooth = v.get("smooth") or [None] * n
    for k, p in enumerate(order):
        sc = "scene" if k == 0 else f"scene{k + 1}"
        if v.get("space") is not None and p == 0:
            fig.add_trace(_mesh3d(go, v["space"].astype(np.int8), sc, color=SPACE, opacity=0.13))
        fig.add_trace(_smooth3d(go, smooth[p]["structure"], smooth[p]["classes"], sc, smooth[p].get("color", (240, 128, 44)))
                      if smooth[p] else _mesh3d(go, v["grids"][p], sc))
        for tip, u, length in v["arrows"]:  # force arrows end where the load enters the part
            tail, neck = tip - u * length, tip - u * 0.45 * length
            fig.add_trace(go.Scatter3d(x=[tail[0], neck[0]], y=[tail[1], neck[1]], z=[tail[2], neck[2]], mode="lines", scene=sc, hoverinfo="skip",
                                       line=dict(color=COLORS[2], width=7), showlegend=False))
            fig.add_trace(go.Cone(x=[tip[0]], y=[tip[1]], z=[tip[2]], u=[u[0]], v=[u[1]], w=[u[2]], anchor="tip", sizemode="absolute",
                                  sizeref=0.45 * length, colorscale=[[0, COLORS[2]], [1, COLORS[2]]], showscale=False, hoverinfo="skip", scene=sc))
    eye, margin = v.get("eye") or default_eye(shape), 0.18 * max(shape)  # room around the part for the arrows
    aspect = 1.4 if n == 3 else 1.9  # panel width over height: three side by side are narrower
    zoom = scene_zoom(shape, eye, aspect)
    # Plotly pads every 3D scene, so a manual aspect ratio well above 1 is what makes the part fill its panel; the same
    # scale on all three axes keeps the voxels cubic
    fig.update_scenes(dragmode="orbit", aspectmode="manual", aspectratio={a: zoom * (n_ + 2 * margin) for a, n_ in zip("xyz", shape)},
                      camera=dict(eye=dict(zip("xyz", eye)), up=dict(x=0, y=0, z=1), projection=dict(type="orthographic")),
                      **{f"{a}axis": dict(visible=False, range=[-margin, n_ + margin]) for a, n_ in zip("xyz", shape)})
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=800 if n == 4 else 420, paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      font=dict(family="Segoe UI, system-ui, sans-serif", size=13, color="#1d2330"), meta=dict(aspect=aspect))  # the page reads the aspect
    if n == 3:
        fig.update_annotations(font_size=12)  # the panel titles
    return fig


# ---------------------------------------------------------------- views of the part-optimisation studio
def design_labels(p):
    """Class grid of a topopt.Problem: 7 design material, 1 fixed solid, 2 supports, 3 loads."""
    g = np.zeros(p.design.shape, np.int8)
    act = p.design != T.VOID
    g[p.design == T.FREE], g[p.design == T.SOLID] = 7, 1
    g[act & p.fixed.any(-1)] = 2
    g[act & (np.abs(p.force).sum(-1) > 0)] = 3
    return g


def load_arrows(p):
    """One arrow per connected group of loaded voxels, along its total force and ending where the load enters the part."""
    loaded = (np.abs(p.force).sum(-1) > 0) & (p.design != T.VOID)
    groups, n = ndimage.label(loaded, np.ones((3, 3, 3)))
    length, arrows = 0.2 * max(p.design.shape), []
    for i in range(1, n + 1):
        m = groups == i
        f = p.force[m].sum(0)
        if np.linalg.norm(f) < 1e-6 * np.abs(p.force).max() * m.sum():  # a self-balancing torque has no single arrow
            continue
        u, pos = f / np.linalg.norm(f), np.argwhere(m) + 0.5
        arrows.append((pos.mean(0) + u * ((pos - pos.mean(0)) @ u).min() - 0.5 * u, u, length))
    return arrows


def studio_view(p, res=None, smooth=True, title="FINAL", reference=None):
    """Panels for a Problem: INITIAL design space, with a Result the FINAL optimised part (smooth surface or voxels),
    and with a reference structure (a boolean grid, from the dataset) the dataset's own solution next to it."""
    base = design_labels(p)
    view = dict(grids=[base], titles=["INITIAL\nfull design space"], shape=p.design.shape, arrows=load_arrows(p), smooth=[None])
    if res is not None:
        final = base.copy()
        final[base == 7] = 0
        final[(base == 7) & res.structure] = 5  # material the optimiser kept, drawn orange
        view["grids"].append(final)
        view["smooth"].append(dict(structure=res.structure, classes=base) if smooth else None)
        view["titles"].append(f"{title}\noptimised, {res.metrics['mass_saved']:.0%} lighter")
        if reference is not None:
            ref = base.copy()
            ref[base == 7] = 0
            ref[(base == 7) & reference] = 4  # the dataset's solution, drawn blue
            view["grids"].append(ref)
            view["smooth"].append(dict(structure=reference, classes=base, color=(59, 125, 216)) if smooth else None)
            view["titles"].append(f"DATASET REFERENCE\n{int(reference.sum()):,} voxels, OptiStruct")
    return view


def fmt_mass(g):
    return f"{g:.0f} g" if g < 1e3 else f"{g / 1e3:.1f} kg" if g < 1e6 else f"{g / 1e6:.1f} t"


def studio_items(p, res, smooth=True, reference=None):
    """Frames for the slider page: the design space being carved at each snapshot (voxels), then the final result (smooth or voxels).
    With a reference structure every frame also shows the dataset's solution."""
    from types import SimpleNamespace
    active, m, items = int((p.design != T.VOID).sum()), res.metrics, []
    for it, grid in res.snapshots:
        frame = SimpleNamespace(structure=grid, metrics=dict(mass_saved=1 - grid.sum() / active))
        items.append((f"Iteration {it}", f"iteration {it} of {len(res.history)}: {grid.sum() / active:.0%} of the mass kept",
                      figure(studio_view(p, frame, smooth=False, title=f"ITERATION {it}", reference=reference))))
    items.append(("Final", f"{fmt_mass(m['mass_initial_g'])} to {fmt_mass(m['mass_final_g'])} ({m['mass_saved']:.0%} lighter), "
                  f"{m['stiffness_retained']:.0%} of the stiffness kept", figure(studio_view(p, res, smooth=smooth, reference=reference))))
    return items


def export_stl_smooth(path, structure, voxel_mm):
    """Write the smoothed surface of a boolean structure as an STL in millimetres (candidate geometry, not checked for printing)."""
    poly = smooth_surface(structure)
    poly.points = poly.points * voxel_mm
    poly.save(path)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AeroTopoML before and after</title>
<style>
body{margin:0;background:#f6f7f9;color:#1d2330;font:15px/1.45 "Segoe UI",system-ui,sans-serif}
main{max-width:1500px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 2px}.sub{color:#5d6675;margin:0 0 12px}
.bar{display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center;margin-bottom:8px}
select,button{font:inherit;padding:6px 10px;border:1px solid #c9cfd8;border-radius:6px;background:#fff;cursor:pointer}
input[type=range]{width:min(420px,60vw);vertical-align:middle}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:#445}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
#plot{background:#fff;border:1px solid #dfe3ea;border-radius:10px;margin-top:8px}
.note{color:#5d6675;font-size:13px;margin-top:8px}
.embed h1,.embed .sub,.embed .note{display:none}.embed main{padding:8px}.embed{background:#fff}.embed #plot{border:0}
</style></head><body class="__BODY__"><main>
<h1>__TITLE__</h1>
<p class="sub">__SUB__</p>
<div class="bar">__PICKER__<b id="stats"></b></div>
<div class="legend">__LEGEND__</div>
<div id="plot"></div>
<p class="note">Drag to rotate (all panels turn together), scroll to zoom, double-click to reset. Research prototype, not a certified design tool.</p>
</main>
<script>__PLOTLYJS__</script>
<script>
const FIGS = __FIGS__, META = __META__;
const gd = document.getElementById('plot'), pick = document.getElementById('pick');
// Plotly ties the scale of a 3D scene to the height of its panel. The preset zoom suits panels about `aspect` times wider than tall
// (the figure says which), so the figure height follows the page width and never exceeds the window: the part then fills its panel
// without being cropped. Inside the Streamlit app the iframe grows with the page, so there the cap is a fixed height instead.
const EMBED = document.body.classList.contains('embed');
function fit() {
  const L = gd._fullLayout, n = Object.keys(L).filter(k => /^scene\\d*$/.test(k)).length, rows = n === 4 ? 2 : 1, cols = n === 4 ? 2 : n;
  const wanted = 40 + rows * (gd.clientWidth / cols) / ((L.meta && L.meta.aspect) || 1.9) / (rows > 1 ? 0.93 : 1);
  const h = Math.round(Math.max(200, Math.min(wanted, EMBED ? (rows > 1 ? 820 : 600) : window.innerHeight - gd.offsetTop - 20)));
  return Math.abs(h - L.height) < 3 ? null : Plotly.relayout(gd, {height: h});
}
function show(i) {
  Plotly.react(gd, FIGS[i].data, FIGS[i].layout, {responsive: true, displaylogo: false}).then(fit);
  document.getElementById('stats').textContent = META[i].stats;
}
const slide = document.getElementById('slide'), play = document.getElementById('play');
if (pick) { META.forEach((m, i) => pick.add(new Option(m.label, i))); pick.onchange = () => show(+pick.value); }
if (slide) {  // frames of the optimisation, the last one is the result
  slide.max = FIGS.length - 1;
  slide.value = FIGS.length - 1;
  slide.oninput = () => show(+slide.value);
  let timer = null;
  play.onclick = () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = '▶ Replay'; return; }
    let i = 0;
    show(0); slide.value = 0; play.textContent = '⏸ Pause';
    timer = setInterval(() => {
      if (++i >= FIGS.length) { clearInterval(timer); timer = null; play.textContent = '▶ Replay'; return; }
      slide.value = i; show(i);
    }, 600);
  };
}
show(slide ? FIGS.length - 1 : 0);
new ResizeObserver(() => Plotly.Plots.resize(gd).then(fit)).observe(document.body);  // redo it when the page or an iframe around it is resized
// keep the cameras of all panels identical: a drag in one panel is copied to the others. Our own update makes Plotly emit
// camera events too, so they are ignored for a while (otherwise the panels keep re-copying each other and drift apart)
let lock = 0, last = null;
const same = (a, b) => b && ['eye', 'up', 'center'].every(k => ['x', 'y', 'z'].every(c => Math.abs(a[k][c] - b[k][c]) < 1e-3));
gd.on('plotly_relayout', ev => {
  const key = Object.keys(ev).find(k => /^scene\\d*\\.camera$/.test(k));
  if (!key || performance.now() < lock || same(ev[key], last)) return;
  last = ev[key];
  const upd = {};
  Object.keys(gd._fullLayout).filter(k => /^scene\\d*$/.test(k) && k + '.camera' !== key).forEach(k => { upd[k + '.camera'] = last; });
  lock = performance.now() + 1500;
  Plotly.relayout(gd, upd).then(() => { lock = performance.now() + 500; });
});
</script></body></html>
"""


SELTO_TITLE = "AeroTopoML: before and after in 3D"
SELTO_SUB = "SELTO disc_simple test samples. Before = reference from conventional topology optimisation, after = PCA + MLP prediction."


def html_page(items, picker=True, embed=False, legend=LEGEND_SELTO, title=SELTO_TITLE, subtitle=SELTO_SUB):
    """Self-contained page. items: list of (label, stats text, plotly figure); the viewer switches between them.
    picker: True or "select" for a drop-down, "slider" for a slider with a play button (frames of an optimisation), False for none.
    embed: drop the page header, for use inside another page (the Streamlit app)."""
    from plotly.offline import get_plotlyjs
    chips = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in legend)
    control = {True: '<label>Test sample <select id="pick"></select></label>', "select": '<label>Part <select id="pick"></select></label>',
               "slider": '<button id="play">▶ Replay</button><input id="slide" type="range" min="0" max="1" value="1">'}.get(picker, "")
    meta = json.dumps([dict(label=lbl, stats=st) for lbl, st, _ in items])
    figs = "[" + ",".join(f.to_json() for _, _, f in items) + "]"
    return (PAGE.replace("__BODY__", "embed" if embed else "").replace("__PICKER__", control).replace("__LEGEND__", chips)
            .replace("__TITLE__", title).replace("__SUB__", subtitle).replace("__PLOTLYJS__", get_plotlyjs())
            .replace("__FIGS__", figs).replace("__META__", meta))


# ---------------------------------------------------------------- static renders and native window (PyVista)
def render(views, path=None, panel_px=(520, 400)):
    """PyVista scene with one row per view and one column per panel. Saves a PNG, or opens a window if path is None."""
    import pyvista as pv
    from matplotlib.colors import ListedColormap
    rows, cols = len(views), max(len(v["grids"]) for v in views)
    pl = pv.Plotter(shape=(rows, cols), off_screen=path is not None, border=False,
                    window_size=(panel_px[0] * cols, panel_px[1] * rows))
    pl.set_background("white")
    cmap = ListedColormap(COLORS)
    for r, v in enumerate(views):
        tip, u, length = v["arrows"][0]
        c0 = tip - u * length
        for c, g in enumerate(v["grids"]):
            pl.subplot(r, c)
            if c == 0:
                verts, quads, _ = voxel_mesh(v["space"].astype(np.int8))
                pl.add_mesh(pv.PolyData.from_regular_faces(verts.astype(float), quads), color=SPACE, opacity=0.13)
            verts, quads, cls = voxel_mesh(g)
            poly = pv.PolyData.from_regular_faces(verts.astype(float), quads)
            poly.cell_data["class"] = cls
            pl.add_mesh(poly, scalars="class", cmap=cmap, clim=(0.5, len(COLORS) + 0.5), show_edges=True, edge_color="#262b33",
                        line_width=0.4, show_scalar_bar=False)
            pl.add_mesh(pv.Arrow(start=c0, direction=u, scale=length, tip_length=0.35, tip_radius=0.12, shaft_radius=0.04), color=COLORS[2])
            pl.add_text(v["titles"][c], position="upper_edge", font_size=9, color="#1d2330")
            if c == 0:
                pl.add_text("dark gray: fixed solid   green: supports   red: load patch and direction", position="lower_edge", font_size=6, color="#5d6675")
            elif c == 3:
                pl.add_text("gray: in both   blue: reference only   orange: prediction only", position="lower_edge", font_size=6, color="#5d6675")
            pl.camera.position, pl.camera.focal_point, pl.camera.up = (19.5, -33.0, 47.0), (19.5, 20.5, 2.0), (0, 0, 1)
            pl.camera.parallel_projection, pl.camera.parallel_scale = True, 21.0
    if path is None:
        pl.link_views()
        pl.show()
        return
    pl.screenshot(path)
    pl.close()


# ---------------------------------------------------------------- command line
def main(argv):
    d = A.load("test")
    pipe, cfg = A.load_pipeline()
    dens = pipe.density(d["x"])
    pred = A.structures(dens, d, **cfg["post"])
    if argv[:1] == ["--show"]:
        i = int(argv[1])
        render([view(d, i, pred[i])])
        return
    m = A.evaluate(dens, d, **cfg["post"])
    out = A.OUT / "3d"
    (out / "stl").mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("3d_test_*.png"), *(out / "stl").glob("test_*.stl")]:  # renders of an earlier model must not linger
        old.unlink()
    order, n = np.argsort(m["dice_free"]), len(pred)
    picks = [int(order[j]) for j in (n - 1, int(0.75 * n), n // 2, int(0.25 * n), 0)]  # best ... worst by free-voxel Dice
    for i in picks:
        render([view(d, i, pred[i])], out / f"3d_test_{i:03d}.png")
        for name, vox in (("reference", d["density"][i]), ("predicted", pred[i])):
            export_stl(out / "stl" / f"test_{i:03d}_{name}.stl", vox, float(d["info"][3]))
    render([view(d, i, pred[i]) for i in picks], out / "gallery.png")
    ids = [int(order[j]) for j in np.linspace(n - 1, 0, 20).astype(int)]
    items = []
    for i in ids:
        phi = np.degrees(np.arctan2(d["centroid"][i][1] - A.AXIS, d["centroid"][i][0] - A.AXIS))
        stats = (f"Dice {m['dice'][i]:.2f} overall, {m['dice_free'][i]:.2f} on free voxels | "
                 f"material voxels: reference {d['density'][i].sum()}, prediction {pred[i].sum()}")
        items.append((f"Test #{i}: load at {phi:.0f} deg, Dice(free) {m['dice_free'][i]:.2f}", stats, figure(view(d, i, pred[i]))))
    html = A.OUT / "before_after_3d.html"
    html.write_text(html_page(items), encoding="utf-8")
    print("PNG/STL:", out, "\nHTML:", html, f"({html.stat().st_size / 1e6:.1f} MB)", "\nsamples:", picks)


if __name__ == "__main__":
    main(sys.argv[1:])
