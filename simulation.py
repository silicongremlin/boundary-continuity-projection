from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.typing import NDArray

# ============================================================
# SIM_PROJECT_4D_V4_SMOOTH_GIF_LABELS_FIXED
# Boundary Continuity + Phase Reconstruction
#
# Default:
#   python sim_project_4d_v4_smooth_gif_labels_fixed.py
#   -> interactive preview
#
# Export:
#   python sim_project_4d_v4_smooth_gif_labels_fixed.py --gif
#   -> saves GIF only
#
# Latent state:
#   (x, y, z, w)
#
# Observable:
#   2D projection: (x, y)
#   3D projection: rotated (x', y', z') with w folded in
# ============================================================

FloatArray = NDArray[np.float64]


# ============================================================
# CLI / MODE
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boundary continuity in projected latent state simulator"
    )

    parser.add_argument(
        "--gif",
        action="store_true",
        help="Export a GIF instead of opening the interactive preview.",
    )

    parser.add_argument(
        "--gif-path",
        default="assets/boundary_continuity_phase_reconstruction.gif",
        help="Output GIF path.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override total frame count.",
    )

    parser.add_argument(
        "--particles",
        type=int,
        default=None,
        help="Override particle count.",
    )

    return parser.parse_args()


args = parse_args()

SAVE_GIF = bool(args.gif)
GIF_PATH = Path(args.gif_path)
GIF_FPS = 20

# Preview defaults
N = 800
TRAIL_LEN = 38
TRAIL_STRIDE = 8
TOTAL_FRAMES = 900
INTERVAL_MS = 25
DRAW_3D_EVERY = 1
DPI = 90

# GIF defaults: smaller so export does not punish your machine.
if SAVE_GIF:
    N = 520
    TRAIL_LEN = 28
    TRAIL_STRIDE = 10
    TOTAL_FRAMES = 260
    INTERVAL_MS = 40
    DRAW_3D_EVERY = 2
    DPI = 85

if args.frames is not None:
    TOTAL_FRAMES = int(args.frames)

if args.particles is not None:
    N = int(args.particles)


# ============================================================
# SEED
# ============================================================
np.random.seed(7)


# ============================================================
# PARTICLES / LATENT 4D STATE
# ============================================================
state4: FloatArray = np.zeros((N, 4), dtype=np.float64)
state4[:, 0] = np.random.uniform(-2.8, 2.8, N)
state4[:, 1] = np.random.uniform(-2.8, 2.8, N)
state4[:, 2] = np.random.normal(0.0, 0.25, N)
state4[:, 3] = np.random.normal(0.0, 0.35, N)

vel4: FloatArray = np.zeros_like(state4)
initial4: FloatArray = state4.copy()

trail_xy: FloatArray = np.repeat(state4[:, :2][None, :, :], TRAIL_LEN, axis=0)


# ============================================================
# PARAMETERS
# ============================================================
dt = 0.032

r0 = 1.25
sigma = 0.22

orth_gain = 0.50
swirl_gain = 1.20
radial_gain = 0.42
couple_gain = 0.90

z_gain = 0.42
w_gain = 0.34

damping = 0.94
escape_radius = 3.15

camera_elev = 28
camera_spin = 0.28

rot_xw_speed = 0.010
rot_yw_speed = 0.007
rot_zw_speed = 0.006

boundary_threshold = 0.35
collapse_distortion_threshold = 1.55
reconstruct_continuity_threshold = 0.68


# ============================================================
# FIGURE
# ============================================================
fig = plt.figure(figsize=(21, 7))
fig.suptitle(
    "Boundary Continuity and Phase Reconstruction — Full Traced Model",
    fontsize=15,
    y=0.98,
)

ax2d = fig.add_subplot(1, 3, 1)
ax3d = fig.add_subplot(1, 3, 2, projection="3d")
ax3d_any = cast(Any, ax3d)
axm = fig.add_subplot(1, 3, 3)


# ============================================================
# 2D VIEW
# ============================================================
ax2d.set_title("Observable Projection + Particle Trails")
ax2d.set_xlim(-3, 3)
ax2d.set_ylim(-3, 3)
ax2d.set_aspect("equal")
ax2d.set_xlabel("observable x")
ax2d.set_ylabel("observable y")

theta: FloatArray = np.linspace(0, 2 * np.pi, 600, dtype=np.float64)

# Boundary rings
ax2d.plot(
    r0 * np.cos(theta),
    r0 * np.sin(theta),
    linewidth=1.25,
    label="boundary core",
)
ax2d.plot(
    (r0 + sigma) * np.cos(theta),
    (r0 + sigma) * np.sin(theta),
    linewidth=0.7,
    linestyle="--",
    label="boundary outer",
)
ax2d.plot(
    (r0 - sigma) * np.cos(theta),
    (r0 - sigma) * np.sin(theta),
    linewidth=0.7,
    linestyle="--",
    label="boundary inner",
)

ax2d.text(
    0.0,
    r0 + 0.35,
    "boundary zone",
    ha="center",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round", alpha=0.18),
)

ax2d.text(
    -2.85,
    -2.75,
    "2D readout: Π(x,y,z,w) → (x,y)",
    ha="left",
    va="bottom",
    fontsize=8,
    alpha=0.8,
)

ax2d.text(
    -2.85,
    -2.55,
    "Particle color: dark/purple = lower latent phase w, yellow = higher latent phase w",
    ha="left",
    va="bottom",
    fontsize=8,
    alpha=0.8,
)

ax2d.text(
    -2.85,
    -2.35,
    "Thin lines = recent particle trajectories",
    ha="left",
    va="bottom",
    fontsize=8,
    alpha=0.8,
)

trail_indices = np.arange(0, N, TRAIL_STRIDE)
trail_lines: list[Any] = []

for _ in trail_indices:
    line, = ax2d.plot([], [], linewidth=0.42, alpha=0.32)
    trail_lines.append(line)

sc2d = ax2d.scatter(
    state4[:, 0],
    state4[:, 1],
    s=8,
    alpha=0.78,
    c=state4[:, 3],
    cmap="viridis",
)

cbar2d = fig.colorbar(sc2d, ax=ax2d, fraction=0.046, pad=0.04)
cbar2d.set_label("latent phase w")

info = ax2d.text(
    0.02,
    0.98,
    "",
    transform=ax2d.transAxes,
    va="top",
    ha="left",
    fontsize=9,
    bbox=dict(boxstyle="round", alpha=0.26),
)

ax2d.legend(loc="lower right", fontsize=8)


# ============================================================
# 3D PROJECTED VIEW
# ============================================================
ax3d.set_title("3D Projection of 4D Latent State")
ax3d.set_xlim(-3, 3)
ax3d.set_ylim(-3, 3)
ax3d.set_zlim(-3, 3)
ax3d.set_xlabel("x′")
ax3d.set_ylabel("y′")
ax3d.set_zlabel("z′ / projected phase")

ring_x: FloatArray = r0 * np.cos(theta)
ring_y: FloatArray = r0 * np.sin(theta)
ring_z: FloatArray = np.zeros_like(theta)

ax3d.plot(ring_x, ring_y, ring_z, linewidth=1.1)

ax3d.text(
    0,
    0,
    0.15,
    "projected boundary",
    fontsize=8,
    ha="center",
)

sc3d = ax3d_any.scatter(
    state4[:, 0],
    state4[:, 1],
    state4[:, 2],
    s=8,
    alpha=0.76,
    c=state4[:, 3],
    cmap="viridis",
)

cbar3d = fig.colorbar(sc3d, ax=ax3d, fraction=0.046, pad=0.08)
cbar3d.set_label("projected latent phase w′")


# ============================================================
# METRICS VIEW
# ============================================================
axm.set_title("Continuity / Distortion / Trace Metrics")
axm.set_xlim(0, TOTAL_FRAMES)
axm.set_ylim(0, 2.25)
axm.set_xlabel("frame")
axm.set_ylabel("normalized score")

line_cont, = axm.plot([], [], label="global continuity")
line_dist, = axm.plot([], [], label="boundary distortion")
line_occ, = axm.plot([], [], label="boundary occupancy")
line_w, = axm.plot([], [], label="latent phase spread |w|")
line_trap, = axm.plot([], [], label="orbit/trap fraction")

# threshold guides
axm.axhline(0.68, linewidth=0.6, linestyle="--", alpha=0.35)
axm.text(5, 0.70, "reconstruction threshold", fontsize=8, alpha=0.7)

axm.axhline(1.55, linewidth=0.6, linestyle="--", alpha=0.35)
axm.text(5, 1.57, "local collapse threshold", fontsize=8, alpha=0.7)

axm.text(
    5,
    2.13,
    "Metric legend:\n"
    "global continuity = radial distribution persistence\n"
    "boundary distortion = local displacement near boundary\n"
    "occupancy = fraction inside boundary zone\n"
    "|w| spread = latent phase magnitude\n"
    "orbit/trap = tangential boundary motion fraction",
    fontsize=8,
    va="top",
    bbox=dict(boxstyle="round", alpha=0.16),
)

axm.legend(loc="upper right")

frames: list[int] = []
continuity_vals: list[float] = []
distortion_vals: list[float] = []
occupancy_vals: list[float] = []
w_vals: list[float] = []
trap_vals: list[float] = []

phase_marker = axm.axvline(0, linewidth=0.8, alpha=0.35)


# ============================================================
# HELPERS
# ============================================================
def boundary_strength(x: FloatArray, y: FloatArray) -> FloatArray:
    r = np.sqrt(x * x + y * y)
    return np.exp(-((r - r0) ** 2) / (2 * sigma**2))


def rotate_4d(points: FloatArray, frame: int) -> FloatArray:
    """
    Rotates latent 4D state before 3D projection.
    Hidden w-phase becomes visible as projected motion.
    """
    p: FloatArray = points.copy()

    ax = rot_xw_speed * frame
    ay = rot_yw_speed * frame
    az = rot_zw_speed * frame

    # x-w rotation
    x = p[:, 0].copy()
    w = p[:, 3].copy()
    p[:, 0] = x * np.cos(ax) - w * np.sin(ax)
    p[:, 3] = x * np.sin(ax) + w * np.cos(ax)

    # y-w rotation
    y = p[:, 1].copy()
    w = p[:, 3].copy()
    p[:, 1] = y * np.cos(ay) - w * np.sin(ay)
    p[:, 3] = y * np.sin(ay) + w * np.cos(ay)

    # z-w rotation
    z = p[:, 2].copy()
    w = p[:, 3].copy()
    p[:, 2] = z * np.cos(az) - w * np.sin(az)
    p[:, 3] = z * np.sin(az) + w * np.cos(az)

    return p


def latent_vector_field(s: FloatArray, frame: int) -> tuple[FloatArray, FloatArray]:
    """
    Computes velocity in latent 4D.

    x,y are observable coordinates.
    z is field/phase height.
    w is hidden latent phase / continuity reserve.
    """
    x = s[:, 0]
    y = s[:, 1]
    z = s[:, 2]
    w = s[:, 3]

    r = np.sqrt(x * x + y * y) + 1e-9
    phi = np.arctan2(y, x)

    B = boundary_strength(x, y)

    er_x = x / r
    er_y = y / r

    et_x = -y / r
    et_y = x / r

    # Orthogonal x/y modes
    u = np.sin(2.8 * x + 0.040 * frame)
    v = np.sin(2.8 * y - 0.035 * frame)

    # Hidden phase mode
    q = np.sin(w + 0.025 * frame)

    # Nonlinear coupling
    coupling_xy = couple_gain * B * u * v
    coupling_w = B * q * (u - v)

    twist = np.sin(3.0 * phi - 0.050 * frame + 0.6 * w)

    tangential = swirl_gain * B * (1.0 + 0.55 * twist + 0.35 * coupling_xy)
    radial = radial_gain * B * np.sin(2.0 * phi + 0.040 * frame + coupling_xy)

    drift_x = orth_gain * u
    drift_y = orth_gain * v

    dx = drift_x + tangential * et_x + radial * er_x
    dy = drift_y + tangential * et_y + radial * er_y

    dz = z_gain * (
        -0.22 * z
        + B * np.sin(2.0 * phi + w + 0.035 * frame)
        + 0.40 * coupling_xy
    )

    dw = w_gain * (
        -0.12 * w
        + coupling_w
        + 0.35 * B * np.cos(3.0 * phi - 0.030 * frame)
    )

    field_vel: FloatArray = np.column_stack([dx, dy, dz, dw]).astype(np.float64)
    return field_vel, B.astype(np.float64)


def global_continuity_score(current: FloatArray, initial: FloatArray) -> float:
    """
    Compare radial distributions in observable x-y.
    High means global observable structure remains similar.
    """
    r_now = np.sqrt(current[:, 0] ** 2 + current[:, 1] ** 2)
    r_ini = np.sqrt(initial[:, 0] ** 2 + initial[:, 1] ** 2)

    hist_now, _ = np.histogram(r_now, bins=44, range=(0, 3), density=True)
    hist_ini, _ = np.histogram(r_ini, bins=44, range=(0, 3), density=True)

    a = hist_now - hist_now.mean()
    b = hist_ini - hist_ini.mean()

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float((np.dot(a, b) / denom + 1.0) / 2.0)


def boundary_distortion_score(current: FloatArray, initial: FloatArray) -> float:
    """
    Average latent displacement for particles near the boundary.
    """
    x = current[:, 0]
    y = current[:, 1]

    B = boundary_strength(x, y)
    mask = B > boundary_threshold

    if int(mask.sum()) == 0:
        return 0.0

    disp = np.linalg.norm(current[mask] - initial[mask], axis=1)
    return float(np.mean(disp))


def orbit_trap_fraction(current: FloatArray, velocity: FloatArray) -> float:
    """
    Estimates how many particles are in boundary-local orbit/trap behavior.
    """
    x = current[:, 0]
    y = current[:, 1]

    vx = velocity[:, 0]
    vy = velocity[:, 1]

    r = np.sqrt(x * x + y * y) + 1e-9
    B = boundary_strength(x, y)

    er_x = x / r
    er_y = y / r
    et_x = -y / r
    et_y = x / r

    radial_v = np.abs(vx * er_x + vy * er_y)
    tangential_v = np.abs(vx * et_x + vy * et_y)

    trapped = (B > boundary_threshold) & (tangential_v > 1.35 * radial_v)
    return float(np.mean(trapped))


def classify_state(cont: float, dist: float, occ: float, trap: float) -> str:
    """
    Human-readable phase label.
    """
    if dist > collapse_distortion_threshold and cont < 0.55:
        return "COLLAPSE / GLOBAL CONTINUITY WEAK"

    if dist > collapse_distortion_threshold and cont >= 0.55:
        return "LOCAL COLLAPSE / GLOBAL CONTINUITY SURVIVES"

    if trap > 0.18:
        return "ORBIT / BOUNDARY TRAP"

    if occ < 0.07 and cont > reconstruct_continuity_threshold:
        return "POST-BOUNDARY RECONSTRUCTION"

    if occ > 0.20:
        return "BOUNDARY TRAVERSAL"

    return "FIELD EVOLUTION"


def respawn_escaped_particles() -> None:
    global state4, vel4, initial4, trail_xy

    r = np.sqrt(state4[:, 0] ** 2 + state4[:, 1] ** 2)
    out = r > escape_radius

    if np.any(out):
        count = int(out.sum())

        state4[out, 0] = np.random.uniform(-2.6, 2.6, count)
        state4[out, 1] = np.random.uniform(-2.6, 2.6, count)
        state4[out, 2] = np.random.normal(0.0, 0.25, count)
        state4[out, 3] = np.random.normal(0.0, 0.35, count)

        vel4[out] = 0.0
        initial4[out] = state4[out]

        # reset trails so respawns do not draw teleport streaks
        for ti in range(TRAIL_LEN):
            trail_xy[ti, out, :] = state4[out, :2]


def update_trails() -> None:
    global trail_xy
    trail_xy[:-1] = trail_xy[1:]
    trail_xy[-1] = state4[:, :2]


# ============================================================
# UPDATE LOOP
# ============================================================
def update(frame: int) -> tuple[Any, ...]:
    global state4, vel4

    field_vel, B = latent_vector_field(state4, frame)

    vel4 = damping * vel4 + dt * field_vel
    state4 = state4 + vel4

    respawn_escaped_particles()
    update_trails()

    projected = rotate_4d(state4, frame)

    cont = global_continuity_score(state4, initial4)
    dist = boundary_distortion_score(state4, initial4)
    occ = float(np.mean(B > boundary_threshold))
    w_spread = float(np.mean(np.abs(state4[:, 3])))
    trap = orbit_trap_fraction(state4, vel4)
    phase_label = classify_state(cont, dist, occ, trap)

    frames.append(frame)
    continuity_vals.append(cont)
    distortion_vals.append(min(dist, 2.25))
    occupancy_vals.append(occ * 2.25)
    w_vals.append(min(w_spread, 2.25))
    trap_vals.append(trap * 2.25)

    # 2D observable projection
    sc2d.set_offsets(state4[:, :2])
    sc2d.set_array(state4[:, 3])

    # 2D trails
    for line, idx in zip(trail_lines, trail_indices):
        xy = trail_xy[:, idx, :]
        line.set_data(xy[:, 0], xy[:, 1])

    # 3D projected view, smoother than delete/recreate
    if frame % DRAW_3D_EVERY == 0:
        setattr(sc3d, "_offsets3d", (projected[:, 0], projected[:, 1], projected[:, 2]))
        sc3d.set_array(projected[:, 3])
        ax3d.view_init(elev=camera_elev, azim=45 + camera_spin * frame)

    # Metrics
    line_cont.set_data(frames, continuity_vals)
    line_dist.set_data(frames, distortion_vals)
    line_occ.set_data(frames, occupancy_vals)
    line_w.set_data(frames, w_vals)
    line_trap.set_data(frames, trap_vals)
    phase_marker.set_xdata([frame, frame])

    info.set_text(
        f"frame = {frame}\n"
        f"state = {phase_label}\n"
        f"global continuity = {cont:.3f}\n"
        f"boundary distortion = {dist:.3f}\n"
        f"boundary occupancy = {occ:.3f}\n"
        f"latent |w| spread = {w_spread:.3f}\n"
        f"orbit/trap fraction = {trap:.3f}\n"
        f"mode = {'GIF EXPORT' if SAVE_GIF else 'INTERACTIVE PREVIEW'}"
    )

    return (
        sc2d,
        sc3d,
        line_cont,
        line_dist,
        line_occ,
        line_w,
        line_trap,
        phase_marker,
        info,
        *trail_lines,
    )


# ============================================================
# RUN
# ============================================================
ani = FuncAnimation(
    fig,
    update,
    frames=TOTAL_FRAMES,
    interval=INTERVAL_MS,
    blit=False,
)

plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

if SAVE_GIF:
    print(f"Saving GIF to: {GIF_PATH.resolve()}")
    writer = PillowWriter(fps=GIF_FPS)
    ani.save(str(GIF_PATH), writer=writer, dpi=DPI)
    plt.close(fig)
    print(f"Saved GIF: {GIF_PATH.resolve()}")
else:
    print("Interactive preview mode.")
    print("To export GIF, run:")
    print(f"  python {Path(__file__).name} --gif")
    plt.show()