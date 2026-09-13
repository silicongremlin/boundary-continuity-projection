from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

# ============================================================
# SIM_PROJECT_4D_CONTROL_NO_BOUNDARY
#
# Control test:
#   REMOVE boundary field.
#
# Keeps:
#   - orthogonal x/y modes
#   - latent z/w state
#   - 4D projection
#   - particle trails
#
# Removes:
#   - boundary-local swirl
#   - boundary-local radial compression
#   - boundary-local nonlinear coupling
#   - boundary occupancy / boundary trap logic
#
# Question:
#   Do the recurring collapse/orbit/reconstruction patterns still appear
#   without an explicit boundary?
# ============================================================

FloatArray = NDArray[np.float64]

np.random.seed(7)

# -----------------------------
# PARTICLES / LATENT 4D STATE
# -----------------------------
N = 800
TRAIL_LEN = 45

state4: FloatArray = np.zeros((N, 4), dtype=np.float64)
state4[:, 0] = np.random.uniform(-2.8, 2.8, N)
state4[:, 1] = np.random.uniform(-2.8, 2.8, N)
state4[:, 2] = np.random.normal(0.0, 0.25, N)
state4[:, 3] = np.random.normal(0.0, 0.35, N)

vel4: FloatArray = np.zeros_like(state4)
initial4: FloatArray = state4.copy()

trail_xy: FloatArray = np.repeat(state4[:, :2][None, :, :], TRAIL_LEN, axis=0)

# -----------------------------
# PARAMETERS
# -----------------------------
dt = 0.032

r0 = 1.25
sigma = 0.22

orth_gain = 0.50
z_gain = 0.42
w_gain = 0.34

damping = 0.94
escape_radius = 3.15

camera_elev = 28
camera_spin = 0.28

rot_xw_speed = 0.010
rot_yw_speed = 0.007
rot_zw_speed = 0.006

# -----------------------------
# FIGURE
# -----------------------------
fig = plt.figure(figsize=(20, 7))

ax2d = fig.add_subplot(1, 3, 1)
ax3d = fig.add_subplot(1, 3, 2, projection="3d")
ax3d_any = cast(Any, ax3d)
axm = fig.add_subplot(1, 3, 3)

# -----------------------------
# 2D VIEW
# -----------------------------
ax2d.set_title("CONTROL: No Boundary — 2D Observable + Trails")
ax2d.set_xlim(-3, 3)
ax2d.set_ylim(-3, 3)
ax2d.set_aspect("equal")
ax2d.set_xlabel("x")
ax2d.set_ylabel("y")

theta: FloatArray = np.linspace(0, 2 * np.pi, 600, dtype=np.float64)

# Faint reference ring only, not used in math
ax2d.plot(r0 * np.cos(theta), r0 * np.sin(theta), linewidth=1.0, alpha=0.25)
ax2d.plot(
    (r0 + sigma) * np.cos(theta),
    (r0 + sigma) * np.sin(theta),
    linewidth=0.5,
    linestyle="--",
    alpha=0.18,
)
ax2d.plot(
    (r0 - sigma) * np.cos(theta),
    (r0 - sigma) * np.sin(theta),
    linewidth=0.5,
    linestyle="--",
    alpha=0.18,
)

trail_stride = 5
trail_indices = np.arange(0, N, trail_stride)
trail_lines: list[Any] = []

for _ in trail_indices:
    line, = ax2d.plot([], [], linewidth=0.45, alpha=0.35)
    trail_lines.append(line)

sc2d = ax2d.scatter(
    state4[:, 0],
    state4[:, 1],
    s=8,
    alpha=0.78,
    c=state4[:, 3],
    cmap="viridis",
)

info = ax2d.text(
    0.02,
    0.98,
    "",
    transform=ax2d.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", alpha=0.25),
)

# -----------------------------
# 3D VIEW
# -----------------------------
ax3d.set_title("CONTROL: No Boundary — 3D Projection of 4D State")
ax3d.set_xlim(-3, 3)
ax3d.set_ylim(-3, 3)
ax3d.set_zlim(-3, 3)
ax3d.set_xlabel("x′")
ax3d.set_ylabel("y′")
ax3d.set_zlabel("z′")

# Faint reference ring only
ring_x: FloatArray = r0 * np.cos(theta)
ring_y: FloatArray = r0 * np.sin(theta)
ring_z: FloatArray = np.zeros_like(theta)
ax3d.plot(ring_x, ring_y, ring_z, linewidth=1.0, alpha=0.25)

sc3d_holder: list[Any] = [
    ax3d_any.scatter(
        state4[:, 0],
        state4[:, 1],
        state4[:, 2],
        s=8,
        alpha=0.76,
        c=state4[:, 3],
        cmap="viridis",
    )
]

# -----------------------------
# METRICS
# -----------------------------
axm.set_title("CONTROL Metrics: No Boundary")
axm.set_xlim(0, 900)
axm.set_ylim(0, 2.25)
axm.set_xlabel("frame")

line_cont, = axm.plot([], [], label="global continuity")
line_dist, = axm.plot([], [], label="global displacement")
line_w, = axm.plot([], [], label="latent phase spread |w|")

axm.legend(loc="upper right")

frames: list[int] = []
continuity_vals: list[float] = []
displacement_vals: list[float] = []
w_vals: list[float] = []

# -----------------------------
# HELPERS
# -----------------------------
def rotate_4d(points: FloatArray, frame: int) -> FloatArray:
    p: FloatArray = points.copy()

    ax = rot_xw_speed * frame
    ay = rot_yw_speed * frame
    az = rot_zw_speed * frame

    x = p[:, 0].copy()
    w = p[:, 3].copy()
    p[:, 0] = x * np.cos(ax) - w * np.sin(ax)
    p[:, 3] = x * np.sin(ax) + w * np.cos(ax)

    y = p[:, 1].copy()
    w = p[:, 3].copy()
    p[:, 1] = y * np.cos(ay) - w * np.sin(ay)
    p[:, 3] = y * np.sin(ay) + w * np.cos(ay)

    z = p[:, 2].copy()
    w = p[:, 3].copy()
    p[:, 2] = z * np.cos(az) - w * np.sin(az)
    p[:, 3] = z * np.sin(az) + w * np.cos(az)

    return p


def latent_vector_field_no_boundary(
    s: FloatArray,
    frame: int,
) -> FloatArray:
    """
    No boundary terms.

    Only orthogonal x/y modes plus weak z/w latent oscillation.
    """
    x = s[:, 0]
    y = s[:, 1]
    z = s[:, 2]
    w = s[:, 3]

    u = np.sin(2.8 * x + 0.040 * frame)
    v = np.sin(2.8 * y - 0.035 * frame)

    # Orthogonal drift only
    dx = orth_gain * u
    dy = orth_gain * v

    # Weak latent response without boundary forcing
    dz = z_gain * (-0.22 * z + 0.15 * np.sin(w + 0.025 * frame))
    dw = w_gain * (-0.12 * w + 0.15 * np.cos(z - 0.020 * frame))

    field_vel: FloatArray = np.column_stack([dx, dy, dz, dw]).astype(np.float64)
    return field_vel


def global_continuity_score(current: FloatArray, initial: FloatArray) -> float:
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


def global_displacement_score(current: FloatArray, initial: FloatArray) -> float:
    disp = np.linalg.norm(current - initial, axis=1)
    return float(np.mean(disp))


def classify_state(cont: float, disp: float) -> str:
    if disp > 1.55 and cont < 0.55:
        return "GLOBAL DISPERSION / CONTINUITY WEAK"
    if disp > 1.55 and cont >= 0.55:
        return "GLOBAL DISPLACEMENT / CONTINUITY SURVIVES"
    if cont > 0.70:
        return "GLOBAL CONTINUITY PRESERVED"
    return "ORTHOGONAL FIELD EVOLUTION"


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

        for ti in range(TRAIL_LEN):
            trail_xy[ti, out, :] = state4[out, :2]


def update_trails() -> None:
    global trail_xy
    trail_xy[:-1] = trail_xy[1:]
    trail_xy[-1] = state4[:, :2]


# -----------------------------
# UPDATE
# -----------------------------
def update(frame: int) -> tuple[Any, ...]:
    global state4, vel4

    field_vel = latent_vector_field_no_boundary(state4, frame)

    vel4 = damping * vel4 + dt * field_vel
    state4 = state4 + vel4

    respawn_escaped_particles()
    update_trails()

    projected = rotate_4d(state4, frame)

    cont = global_continuity_score(state4, initial4)
    disp = global_displacement_score(state4, initial4)
    w_spread = float(np.mean(np.abs(state4[:, 3])))
    phase_label = classify_state(cont, disp)

    frames.append(frame)
    continuity_vals.append(cont)
    displacement_vals.append(min(disp, 2.25))
    w_vals.append(min(w_spread, 2.25))

    sc2d.set_offsets(state4[:, :2])
    sc2d.set_array(state4[:, 3])

    for line, idx in zip(trail_lines, trail_indices):
        xy = trail_xy[:, idx, :]
        line.set_data(xy[:, 0], xy[:, 1])

    sc3d_holder[0].remove()
    sc3d_holder[0] = ax3d_any.scatter(
        projected[:, 0],
        projected[:, 1],
        projected[:, 2],
        s=8,
        alpha=0.76,
        c=projected[:, 3],
        cmap="viridis",
    )

    ax3d.view_init(elev=camera_elev, azim=45 + camera_spin * frame)

    line_cont.set_data(frames, continuity_vals)
    line_dist.set_data(frames, displacement_vals)
    line_w.set_data(frames, w_vals)

    info.set_text(
        f"frame = {frame}\n"
        f"state = {phase_label}\n"
        f"global continuity = {cont:.3f}\n"
        f"global displacement = {disp:.3f}\n"
        f"latent |w| spread = {w_spread:.3f}\n"
        f"CONTROL = NO BOUNDARY"
    )

    return (
        sc2d,
        sc3d_holder[0],
        line_cont,
        line_dist,
        line_w,
        info,
        *trail_lines,
    )


ani = FuncAnimation(
    fig,
    update,
    frames=900,
    interval=25,
    blit=False,
)

plt.tight_layout()
plt.show()