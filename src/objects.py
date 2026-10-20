"""Library of real-world-inspired parts as voxel design problems (design space, supports, loads, material).

Every shape is written in design units, so the 'detail' setting only changes how finely it is voxelised. The parts are
simplified scale models for demonstration: the loads are plausible, not certified design loads.
"""
from dataclasses import dataclass

import numpy as np

import topopt as T

DETAIL = {"Fast": 0.75, "Normal": 1.0, "Fine": 1.3}  # voxels per design unit
FASTENERS, AERO, AUTO, MACHINES, STRUCTURES, DATASETS = ("Fasteners & tools", "Aerospace & drones", "Automotive", "Machines & lifting",
                                                         "Structures & furniture", "Real dataset examples (SELTO)")
STEEL, ALUMINIUM, NYLON, CONCRETE, OAK = (210_000.0, 7.85), (70_000.0, 2.7), (1_700.0, 1.01), (30_000.0, 2.4), (11_000.0, 0.75)  # Young's modulus MPa, density g/cm3


class Canvas:
    """Voxel canvas in design units: coordinates X, Y, Z of the voxel centres and the arrays that make a Problem."""

    def __init__(self, base, s):
        self.dims = tuple(max(3, round(b * s)) for b in base)
        # voxel centres always span the full design extent, whatever the detail
        self.X, self.Y, self.Z = np.meshgrid(*[(np.arange(n) + 0.5) / n * b for n, b in zip(self.dims, base)], indexing="ij")
        self.design = np.full(self.dims, T.VOID, np.int8)
        self.fixed = np.zeros((*self.dims, 3), bool)
        self.force = np.zeros((*self.dims, 3))

    def free(self, m):
        self.design[m] = T.FREE

    def solid(self, m):
        self.design[m] = T.SOLID

    def support(self, m, axes=(0, 1, 2)):
        self.design[m] = T.SOLID
        for a in axes:
            self.fixed[m, a] = True

    def load(self, m, direction, newtons):
        """Spread newtons over the voxels of m along direction, and keep them as solid."""
        self.design[m] = T.SOLID
        self.force[m] += np.asarray(direction, float) / np.linalg.norm(direction) * newtons / max(int(m.sum()), 1)


def hexagon(U, V, R):
    """Regular hexagon with corner radius R and corners on the U axis."""
    return (np.abs(V) <= R * 0.866025) & (np.abs(U) * 0.866025 + np.abs(V) * 0.5 <= R * 0.866025)


def nut(c):
    U, V = c.X - 14, c.Y - 14
    r, body = np.hypot(U, V), hexagon(U, V, 13.8)
    c.free(body & (r >= 5.5))  # the bore stays open for the bolt
    thread = body & (r >= 5.5) & (r < 7.2)
    c.load(thread, (0, 0, -1), 20_000)  # the bolt pulls its thread down onto the nut
    c.support(body & (c.Z < 1.2) & (r >= 8.5))  # bearing face on the washer


def bolt(c):
    U, V = c.X - 13, c.Y - 13
    r = np.hypot(U, V)
    head, shank = hexagon(U, V, 12.5) & (c.Z >= 32), (r <= 6.5) & (c.Z < 32)
    c.free(head | shank)
    c.solid(shank & (r <= 2.5))  # core of the shank stays, so the part cannot fall apart
    c.support(head & (c.Z < 33.2) & (r >= 7.8))  # underside of the head rests on the clamped plates
    thread = shank & (c.Z < 9) & (r >= 3.5)
    c.load(thread, (0, 0, -1), 15_000)  # the nut pulls the threaded end down: tension in the shank
    c.load(thread, (1, 0, 0), 5_000)  # and a sideways load that bends it
    flats = head & ~hexagon(U, V, 10.4) & (c.Z >= 34)  # the outer skin of the head, where a wrench grips the six flats
    t = np.stack([-V, U], -1)[flats]
    c.design[flats] = T.SOLID
    c.force[flats, :2] += t / np.linalg.norm(t, axis=1, keepdims=True) * 6_000 / flats.sum()  # tightening torque: a ring of tangential forces


def drone(c):
    U, V = c.X - 24, c.Y - 24
    rings = [(sx * 18, sy * 18) for sx in (1, -1) for sy in (1, -1)]  # four motor mounts
    dist = [np.hypot(U - cu, V - cv) for cu, cv in rings]
    plate = (np.abs(U) <= 22) & (np.abs(V) <= 22) & (np.hypot(U, V) <= 28)
    c.free((plate | np.any([d <= 4.8 for d in dist], axis=0)) & ~np.any([d < 2.0 for d in dist], axis=0))  # minus the shaft holes
    for k, ((cu, cv), d) in enumerate(zip(rings, dist)):
        ring = (d >= 2.0) & (d <= 4.8)
        c.load(ring, (0, 0, 1), 12)  # rotor thrust lifts the mount
        t = np.stack([-(V - cv), U - cu], -1)[ring]  # direction around the motor axis
        c.force[ring, :2] += (-1) ** k * t / np.linalg.norm(t, axis=1, keepdims=True) * 3.0 / ring.sum()  # reaction torque of the rotor
    c.support((np.abs(U) <= 5) & (np.abs(V) <= 5))  # battery and electronics deck: carries the weight


def chassis(c):
    X, Y, Z = c.X, c.Y, c.Z
    c.free(np.ones_like(X, bool))
    for x0 in (9, 51):  # four suspension mounts, one per wheel
        for y0 in (1.5, 22.5):
            c.support((np.abs(X - x0) <= 2.5) & (np.abs(Y - y0) <= 1.5) & (Z < 3))
    top = Z >= 5
    c.load(top & (X >= 12) & (X <= 20) & (np.abs(Y - 12) <= 4), (0, 0, -1), 4_000)  # engine
    c.load(top & (X >= 26) & (X <= 40) & (np.abs(Y - 12) <= 8), (0, 0, -1), 6_000)  # cabin and passengers
    c.load(top & (X >= 46) & (X <= 56) & (np.abs(Y - 12) <= 7), (0, 0, -1), 5_000)  # cargo


def control_arm(c):
    X, Y = c.X, c.Y
    half = 18 - 12 * X / 46  # wide at the chassis end, narrow at the wheel end
    holes = [np.hypot(X - 5, Y - y0) < 1.6 for y0 in (6, 34)] + [np.hypot(X - 41, Y - 20) < 1.6]
    c.free((np.abs(Y - 20) <= half) & ~np.any(holes, axis=0))
    for y0 in (6, 34):  # two bushings on the chassis
        d = np.hypot(X - 5, Y - y0)
        c.support((d >= 1.6) & (d <= 3.6))
    ball = (np.hypot(X - 41, Y - 20) >= 1.6) & (np.hypot(X - 41, Y - 20) <= 3.6)  # ball joint to the wheel hub
    c.load(ball, (0, -1, 0), 4_000)  # cornering force
    c.load(ball, (0, 0, 1), 1_500)  # bump


def bracket(c):
    X, Y, Z = c.X, c.Y, c.Z
    c.free(~((X > 10) & (Z > 11)))  # wall plate with an arm, an L shape
    d = np.hypot(X - 27.5, Z - 5.5)
    c.design[(d < 2.2)] = T.VOID  # pin hole
    for yb in (4, 16):  # four bolts into the wall
        for zb in (4, 16):
            c.support((X < 1.5) & (np.hypot(Y - yb, Z - zb) <= 2.6))
    c.load((d >= 2.2) & (d <= 4.5) & (np.abs(Y - 10) <= 5), (0, 0, -1), 1_500)  # a hanging load on the pin


def bridge(c):
    X, Y, Z = c.X, c.Y, c.Z
    c.free(np.ones_like(X, bool))
    c.support((X < 2) & (Z < 2))  # pinned at the left abutment
    c.support((X >= 54) & (Z < 2), axes=(1, 2))  # rolling at the right abutment
    c.load(Z >= 12, (0, 0, -1), 210_000)  # pedestrian crowd on the deck


def hook(c):
    X, Y = c.X, c.Y
    U, V = X - 22, Y - 20
    r, angle = np.hypot(U, V), np.degrees(np.arctan2(V, U))
    c.free(((r >= 9) & (r <= 18) & ~((angle > -60) & (angle < -10))) | ((np.abs(U) <= 6) & (Y >= 30)))  # a C-shaped bowl, a shank above
    c.support((np.abs(U) <= 6) & (Y >= 49))  # the crane holds the hook here
    c.load((r >= 9) & (r <= 11.5) & (angle >= -108) & (angle <= -72), (0, -1, 0), 20_000)  # the load rests in the bowl


def spanner(c):
    X, Y = c.X, c.Y
    U, V = X - 11, Y - 11
    hole = hexagon(U, V, 5.6)
    c.free(((np.hypot(U, V) <= 10.5) | ((X >= 10) & (np.abs(Y - 11) <= 5))) & ~hole)  # round head with a hex opening, and a handle
    c.support(hexagon(U, V, 7.8) & ~hole)  # the jaws grip the nut
    c.load((X >= 67) & (np.abs(Y - 11) <= 5), (0, -1, 0), 400)  # a hand pushes the end of the handle


def conrod(c):
    X, Y = c.X, c.Y
    big, small = np.hypot(X - 11, Y - 11), np.hypot(X - 55, Y - 11)
    shank = (X >= 10) & (X <= 55) & (np.abs(Y - 11) <= 6)
    c.free(((big <= 10) | (small <= 7) | shank) & (big >= 6) & (small >= 3.2))  # two eyes joined by a shank, bores open
    c.support((big >= 6) & (big <= 7.6))  # the crank pin holds the big end
    c.load((small >= 3.2) & (small <= 4.8), (0.96, 0.28, 0), 15_000)  # the piston pin pulls the small end, with a sideways whip


def wing_rib(c):
    X, Y = c.X, c.Y
    s = X / 84
    half = 5 * 0.26 * 84 * (0.2969 * np.sqrt(s) - 0.1260 * s - 0.3516 * s**2 + 0.2843 * s**3 - 0.1015 * s**4)  # NACA 4-digit thickness, 26%
    body = np.abs(Y - 11) <= half
    c.free(body)
    for xs in (21, 57):  # front and rear spar
        c.support(body & (np.abs(X - xs) <= 1.5))
    skin = body & (np.abs(Y - 11) > half - 2.4)  # thick enough not to bend between the web's supports
    c.solid(skin)
    c.load(skin & (X > 21) & (X < 57), (0, 1, 0), 9_000)  # air pressure on the skin between the spars


def pedal(c):
    X, Y = c.X, c.Y
    a, b = np.array([10.0, 10.0]), np.array([28.0, 62.0])  # pivot and foot pad
    t = np.clip(((X - a[0]) * (b[0] - a[0]) + (Y - a[1]) * (b[1] - a[1])) / np.dot(b - a, b - a), 0, 1)
    d = np.hypot(X - (a[0] + t * (b[0] - a[0])), Y - (a[1] + t * (b[1] - a[1])))
    pivot, rod = np.hypot(X - 10, Y - 10), np.hypot(X - 16, Y - 30)
    body = d <= 9 - 4 * t  # a lever that narrows towards the pad
    c.free(body & (pivot >= 2) & (rod >= 1.6))
    c.support((pivot >= 2) & (pivot <= 4.6) & body)  # pivot pin
    c.support((rod >= 1.6) & (rod <= 3.8) & body & (np.abs(Y - 30) <= 1.5), axes=(0,))  # push rod: pushes along x on a line, so no couple
    c.load((np.abs(X - 28) <= 7) & (Y >= 58) & body, (-0.95, -0.3, 0), 600)  # the foot on the pad


def tower(c):
    X, Y, Z = c.X, c.Y, c.Z
    half = 8 - 4.4 * Z / 48  # narrower towards the top
    body = (np.abs(X - 8) <= half) & (np.abs(Y - 8) <= half)
    c.free(body)
    c.support(body & (Z < 2))  # foundation
    c.load(body & (Z >= 46), (1, 0, 0), 30_000)  # wind and cable pull at the top
    c.load(body & (Z >= 46), (0, 0, -1), 10_000)  # weight of the equipment


def stool(c):
    X, Y, Z = c.X, c.Y, c.Z
    c.free(np.ones_like(X, bool))
    for x0 in (2, 18):
        for y0 in (2, 18):
            c.support((np.abs(X - x0) <= 1.6) & (np.abs(Y - y0) <= 1.6) & (Z < 2))  # four feet on the floor
    c.load(Z >= 22, (0, 0, -1), 1_500)  # a person sits on the seat


@dataclass
class Spec:
    key: str
    label: str
    icon: str
    blurb: str
    facts: list
    base: tuple  # voxels at detail 1.0
    unit_mm: float
    material: tuple
    material_name: str
    keep: float
    radius: float
    build: callable
    group: str = ""
    dataset: str = ""  # name of the processed SELTO split this problem comes from, for the dataset entries


CATALOG = {s.key: s for s in [
    Spec("nut", "Hex nut", "🔩", "A hex nut on a bolt: the thread pushes down, the washer face pushes back.",
         ["Thread ring loaded with 20 kN, pulling down", "Bearing face on the washer is fixed", "The bore and the thread ring are kept"],
         (28, 28, 14), 1.0, STEEL, "steel", 0.5, 1.25, nut, FASTENERS),
    Spec("bolt", "Hex bolt", "🔧", "A bolt in tension with a sideways load: how much head and shank does it really need?",
         ["Threaded end loaded with 15 kN tension and 5 kN sideways", "Underside of the head rests on the plates (fixed)", "A thin core of the shank is kept"],
         (26, 26, 42), 1.5, STEEL, "steel", 0.6, 1.25, bolt, FASTENERS),
    Spec("spanner", "Ring spanner", "🔧", "A ring spanner: its hex opening grips a nut and a hand pushes the end of the handle.",
         ["400 N pushing the end of the handle", "The faces of the hex opening are fixed", "The hex opening and the handle end are kept"],
         (72, 22, 5), 3.0, STEEL, "steel", 0.4, 1.25, spanner, FASTENERS),
    Spec("drone", "Drone frame", "🚁", "Quadcopter frame plate: four motor mounts lifting a central electronics deck.",
         ["Rotor thrust 12 N on each motor mount, plus rotor torque", "The central deck is fixed (it carries the weight)", "Motor mounts and the deck are kept"],
         (48, 48, 3), 2.8, NYLON, "3D-printed nylon", 0.5, 1.25, drone, AERO),
    Spec("rib", "Aircraft wing rib", "✈️", "A wing rib seen from the side: air pressure acts on the skin and the two spars hold the rib.",
         ["9 kN of air pressure on the skin between the spars, pushing up", "The front and rear spar are fixed", "The skin and the spars are kept"],
         (84, 22, 4), 10.0, ALUMINIUM, "aluminium", 0.6, 1.25, wing_rib, AERO),
    Spec("chassis", "Car chassis", "🚗", "Ladder-style car chassis: engine, cabin and cargo loads carried to the four wheels.",
         ["Engine 4 kN, cabin 6 kN, cargo 5 kN pushing down", "The four suspension mounts are fixed", "Load patches on the top layer are kept"],
         (60, 24, 6), 80.0, STEEL, "steel", 0.2, 1.25, chassis, AUTO),
    Spec("arm", "Suspension control arm", "🛠️", "A-arm that links the wheel hub to the car body: two bushings, one ball joint.",
         ["Ball joint loaded with 4 kN cornering force and 1.5 kN bump", "Two chassis bushings are fixed", "The bushing and ball-joint eyes are kept"],
         (46, 40, 5), 5.0, ALUMINIUM, "aluminium", 0.3, 1.25, control_arm, AUTO),
    Spec("pedal", "Brake pedal", "🦶", "A pedal lever: it turns on a pivot, the foot pushes the pad and a push rod pushes back.",
         ["600 N from the foot on the pad", "The pivot pin is fixed, the push rod resists in one direction", "The pivot and push-rod eyes are kept"],
         (40, 70, 5), 3.5, STEEL, "steel", 0.5, 1.25, pedal, AUTO),
    Spec("conrod", "Connecting rod", "🔗", "An engine connecting rod: the piston pin pulls the small end, the crank pin holds the big end.",
         ["15 kN pulling the small end, with a sideways whip", "The bore of the big end is fixed", "Both eyes are kept"],
         (64, 22, 6), 3.0, STEEL, "steel", 0.45, 1.25, conrod, AUTO),
    Spec("hook", "Crane hook", "🪝", "A crane hook: it hangs from the crane and the load rests in the bowl.",
         ["20 kN resting in the bowl", "The top of the shank is fixed", "The bowl contact and the fixed top are kept"],
         (44, 52, 6), 8.0, STEEL, "steel", 0.5, 1.25, hook, MACHINES),
    Spec("bracket", "Wall bracket", "📐", "Wall-mounted bracket that holds a hanging load on a pin.",
         ["1.5 kN hanging load on the pin", "Four wall bolts are fixed", "The pin boss and the bolt contact areas are kept"],
         (32, 20, 20), 3.0, ALUMINIUM, "aluminium", 0.3, 1.25, bracket, MACHINES),
    Spec("bridge", "Footbridge", "🌉", "Pedestrian footbridge: a deck on two abutments, everything below it is up for grabs.",
         ["Crowd load of 5 kPa on the deck (210 kN in total)", "Pinned at the left end, rolling at the right end", "The deck is kept"],
         (56, 12, 14), 250.0, CONCRETE, "reinforced concrete", 0.4, 1.25, bridge, STRUCTURES),
    Spec("tower", "Lattice tower", "🗼", "A transmission-style tower: wind and cable pull at the top, the foundation holds it down.",
         ["30 kN sideways and 10 kN of weight at the top", "The foundation layer is fixed", "The top platform and the foundation are kept"],
         (16, 16, 48), 250.0, STEEL, "steel", 0.35, 1.25, tower, STRUCTURES),
    Spec("stool", "Stool", "🪑", "A stool: a person sits on the seat and four feet stand on the floor.",
         ["1.5 kN on the seat", "The four feet are fixed", "The seat is kept"],
         (20, 20, 24), 20.0, OAK, "oak", 0.3, 1.25, stool, STRUCTURES),
    Spec("wheel", "Car wheel", "🛞", "Wheel disc from the SELTO dataset: a hub bolted in the middle, a load on the rim. "
         "The only part with a trained ML model: it answers in a fraction of a millisecond.",
         ["3 kN load on the rim, direction and position taken from a test sample", "The hub is fixed", "Mass to keep = the reference solution's"],
         (39, 39, 4), 2.5, ALUMINIUM, "aluminium", 0.14, 1.25, None, AUTO, "test"),
    Spec("disc_complex", "Disc, two loads", "💿", "A disc from the SELTO dataset like the wheel, but with two loads on the rim: real data, not made here.",
         ["Two loads on the rim, taken from a dataset sample", "The hub is fixed", "Mass to keep = the dataset reference's"],
         None, 2.5, ALUMINIUM, "aluminium", 0.14, 1.25, None, DATASETS, "disc_complex"),
    Spec("sphere_simple", "Sphere, one load", "🔵", "A spherical design space from the SELTO dataset (39 × 39 × 21 voxels): real data, not made here.",
         ["One load, taken from a dataset sample", "The supports are those of the dataset", "Mass to keep = the dataset reference's"],
         None, 2.5, ALUMINIUM, "aluminium", 0.1, 1.25, None, DATASETS, "sphere_simple"),
    Spec("sphere_complex", "Sphere, two loads", "🟣", "The same spherical design space with two loads: the hardest SELTO case.",
         ["Two loads, taken from a dataset sample", "The supports are those of the dataset", "Mass to keep = the dataset reference's"],
         None, 2.5, ALUMINIUM, "aluminium", 0.1, 1.25, None, DATASETS, "sphere_complex"),
]}


def dataset_problem(d, i, name, load_n=3_000.0):
    """The SELTO problem of sample i (arrays of a split from data/processed) as a Problem: its supports and loads.
    The mass to keep is that of the reference structure, so every engine is asked for the same amount of material."""
    design = d["design"][i].astype(np.int8)
    force = np.zeros((*design.shape, 3))
    rows = d["force_idx"][:, 0] == i
    force[tuple(d["force_idx"][rows, 1:].astype(int).T)] = d["force_val"][rows]
    force *= load_n / np.linalg.norm(force, axis=-1).sum()  # the stored values are force densities: only their pattern is used
    return T.Problem(name, design, d["dirichlet"].astype(bool), force, voxel_mm=float(d["info"][3]) * 1000,
                     young_mpa=float(d["info"][0]) / 1e6, density_g_cm3=ALUMINIUM[1],
                     keep=float(d["density"][i].sum()) / float((design != T.VOID).sum()), radius=1.25)


def build(key, detail="Normal", keep=None):
    """The voxel Problem of a catalog part at the chosen detail, optionally with another mass to keep."""
    spec, s = CATALOG[key], DETAIL[detail]
    c = Canvas(spec.base, s)
    spec.build(c)
    return T.Problem(spec.label, c.design, c.fixed, c.force, voxel_mm=spec.unit_mm * spec.base[0] / c.dims[0], young_mpa=spec.material[0],
                     density_g_cm3=spec.material[1], keep=spec.keep if keep is None else keep, radius=spec.radius)
