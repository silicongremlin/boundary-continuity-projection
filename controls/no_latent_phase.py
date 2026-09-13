from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

# ============================================================
# SIM_PROJECT_4D_CONTROL_NO_W_PHASE
#
# Control test:
#   REMOVE active latent w-phase.
#
# Keeps:
#   - orthogonal x/y modes
#   - circular boundary
#   - nonlinear u*v coupling
#   - z phase height
#   - 4D-shaped state container for projection compatibility
#   - particle trails
#
# Removes:
#   - active w dynamics
#   - w-driven twist
#   - w as hidden continuity reserve
#
# Question:
#   Does the full model's behavior depend on latent w-phase,
#   or does x/y/z + boundary + coupling already produce it?
# ============================================================

FloatArray = NDArray[np.float64]

np.random.seed(7)

# -----------------------------
# PARTICLES / STATE
# -----------------------------
N = 800
TRAIL_LEN = 45

state4: FloatArray = np.zeros((N, 4), dtype=np.float64)
state4[:, 0] = np.random.uniform(-2.8, 2.8, N)  # x
state4[:, 1] = np.random.uniform(-2.8, 2.8, N)  # y
state4[:, 2] = np.random.normal(0.0, 0.25, N)   # z / phase height
state4[:, 3] = 0.0                              # w disabled

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
swirl_gain = 1.20
radial_gain = 0.42
couple_gain = 0.90

z_gain = 0.42

damping = 0.94
escape_radius = 3.15

camera_elev = 28
camera_spin = 0.28

# 4D projection rotation speeds
# w is disabled, but keep these as no-op compatible.
rot_xw_speed = 0.010
rot_yw_speed = 0.007
rot_zw_speed = 0.006

boundary_threshold = 0.35
collapse_distortion_threshold = 1.55
reconstruct_continuity_threshold = 0.68

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
ax2d.set_title("CONTROL: No Active w-Phase")
ax2d.set_xlim(-3, 3)
ax2d.set_ylim(-3, 3)
ax2d.set_aspect("equal")
ax2d.set_xlabel("x")
ax2d.set_ylabel("y")

theta: FloatArray = np.linspace(0, 2 * np.pi, 600, dtype=np.float64)

ax2d.plot(r0 * np.cos(theta), r0 * np.sin(theta), linewidth=1.2)
ax2d.plot(
    (r0 + sigma) * np.cos(theta),
    (r0 + sigma) * np.sin(theta),
    linewidth=0.6,
    linestyle="--",
)
ax2d.plot(
    (r0 - sigma) * np.cos(theta),
    (r0 - sigma) * np.sin(theta),
    linewidth=0.6,
    linestyle="--",
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
    c=state4[:, 2],
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
ax3d.set_title("3D Projection: No Active w-Phase")
ax3d.set_xlim(-3, 3)
ax3d.set_ylim(-3, 3)
ax3d.set_zlim(-3, 3)
ax3d.set_xlabel("x′")
ax3d.set_ylabel("y′")
ax3d.set_zlabel("z′")

ring_x: FloatArray = r0 * np.cos(theta)
ring_y: FloatArray = r0 * np.sin(theta)
ring_z: FloatArray = np.zeros_like(theta)
ax3d.plot(ring_x, ring_y, ring_z, linewidth=1.0)

sc3d_holder: list[Any] = [
    ax3d_any.scatter(
        state4[:, 0],
        state4[:, 1],
        state4[:, 2],
        s=8,
        alpha=0.76,
        c=state4[:, 2],
        cmap="viridis",
    )
]

# -----------------------------
# METRICS
# -----------------------------
axm.set_title("CONTROL Metrics: No Active w-Phase")
axm.set_xlim(0, 900)
axm.set_ylim(0, 2.25)
axm.set_xlabel("frame")

line_cont, = axm.plot([], [], label="global continuity")
line_dist, = axm.plot([], [], label="boundary distortion")
line_occ, = axm.plot([], [], label="boundary occupancy")
line_z, = axm.plot([], [], label="z phase spread")
line_trap, = axm.plot([], [], label="orbit/trap fraction")

axm.legend(loc="upper right")

frames: list[int] = []
continuity_vals: list[float] = []
distortion_vals: list[float] = []
occupancy_vals: list[float] = []
z_vals: list[float] = []
trap_vals: list[float] = []

# -----------------------------
# HELPERS
# -----------------------------
def boundary_strength(x: FloatArray, y: FloatArray) -> FloatArray:
    r = np.sqrt(x * x + y * y)
    return np.exp(-((r - r0) ** 2) / (2 * sigma**2))


def rotate_4d(points: FloatArray, frame: int) -> FloatArray:
    """
    4D projection retained for comparable visualization.
    w remains zero/inactive, so projection should not introduce hidden w structure.
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


def latent_vector_field_no_w_phase(
    s: FloatArray,
    frame: int,
) -> tuple[FloatArray, FloatArray]:
    x = s[:, 0]
    y = s[:, 1]
    z = s[:, 2]

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

    # Nonlinear coupling remains, but no w contribution.
    coupling_xy = couple_gain * B * u * v

    # Twist remains boundary/phi-driven only.
    twist = np.sin(3.0 * phi - 0.050 * frame)

    tangential = swirl_gain * B * (1.0 + 0.55 * twist + 0.35 * coupling_xy)
    radial = radial_gain * B * np.sin(2.0 * phi + 0.040 * frame + coupling_xy)

    drift_x = orth_gain * u
    drift_y = orth_gain * v

    dx = drift_x + tangential * et_x + radial * er_x
    dy = drift_y + tangential * et_y + radial * er_y

    dz = z_gain * (
        -0.22 * z
        + B * np.sin(2.0 * phi + 0.035 * frame)
        + 0.40 * coupling_xy
    )

    dw = np.zeros_like(dx)

    field_vel: FloatArray = np.column_stack([dx, dy, dz, dw]).astype(np.float64)
    return field_vel, B.astype(np.float64)


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


def boundary_distortion_score(current: FloatArray, initial: FloatArray) -> float:
    x = current[:, 0]
    y = current[:, 1]

    B = boundary_strength(x, y)
    mask = B > boundary_threshold

    if int(mask.sum()) == 0:
        return 0.0

    disp = np.linalg.norm(current[mask] - initial[mask], axis=1)
    return float(np.mean(disp))


def orbit_trap_fraction(current: FloatArray, velocity: FloatArray) -> float:
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
    if dist > collapse_distortion_threshold and cont < 0.55:
        return "LOCAL COLLAPSE / GLOBAL CONTINUITY WEAK"
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
        state4[out, 3] = 0.0

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

    field_vel, B = latent_vector_field_no_w_phase(state4, frame)

    vel4 = damping * vel4 + dt * field_vel
    state4 = state4 + vel4

    # Hard clamp w inactive.
    state4[:, 3] = 0.0
    vel4[:, 3] = 0.0

    respawn_escaped_particles()
    update_trails()

    projected = rotate_4d(state4, frame)

    cont = global_continuity_score(state4, initial4)
    dist = boundary_distortion_score(state4, initial4)
    occ = float(np.mean(B > boundary_threshold))
    z_spread = float(np.mean(np.abs(state4[:, 2])))
    trap = orbit_trap_fraction(state4, vel4)
    phase_label = classify_state(cont, dist, occ, trap)

    frames.append(frame)
    continuity_vals.append(cont)
    distortion_vals.append(min(dist, 2.25))
    occupancy_vals.append(occ * 2.25)
    z_vals.append(min(z_spread, 2.25))
    trap_vals.append(trap * 2.25)

    sc2d.set_offsets(state4[:, :2])
    sc2d.set_array(state4[:, 2])

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
        c=state4[:, 2],
        cmap="viridis",
    )

    ax3d.view_init(elev=camera_elev, azim=45 + camera_spin * frame)

    line_cont.set_data(frames, continuity_vals)
    line_dist.set_data(frames, distortion_vals)
    line_occ.set_data(frames, occupancy_vals)
    line_z.set_data(frames, z_vals)
    line_trap.set_data(frames, trap_vals)

    info.set_text(
        f"frame = {frame}\n"
        f"state = {phase_label}\n"
        f"global continuity = {cont:.3f}\n"
        f"boundary distortion = {dist:.3f}\n"
        f"boundary occupancy = {occ:.3f}\n"
        f"z phase spread = {z_spread:.3f}\n"
        f"orbit/trap fraction = {trap:.3f}\n"
        f"CONTROL = NO ACTIVE w-PHASE"
    )

    return (
        sc2d,
        sc3d_holder[0],
        line_cont,
        line_dist,
        line_occ,
        line_z,
        line_trap,
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