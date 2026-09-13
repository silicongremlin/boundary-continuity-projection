from typing import Any, cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

# ============================================================
# SIM_PROJECT_4D_CONTROL_RANDOM_BOUNDARY
#
# Control test:
#   Replace clean circular boundary with randomized boundary field.
#
# Keeps:
#   - orthogonal x/y modes
#   - nonlinear u*v coupling
#   - active latent w-phase
#   - z phase height
#   - 4D projection
#   - particle trails
#
# Changes:
#   - boundary B(x,y) is no longer a clean ring
#   - boundary is a randomized smooth constraint field
#
# Question:
#   Does the ejection/admission/reconstruction behavior depend on
#   clean circular topology, or does any boundary-like constraint
#   produce similar behavior?
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

# Random boundary parameters
NUM_BLOBS = 14
blob_centers: FloatArray = np.random.uniform(-2.35, 2.35, size=(NUM_BLOBS, 2)).astype(np.float64)
blob_sigmas: FloatArray = np.random.uniform(0.22, 0.55, size=NUM_BLOBS).astype(np.float64)
blob_weights: FloatArray = np.random.uniform(0.45, 1.20, size=NUM_BLOBS).astype(np.float64)

# Add a few ring-adjacent blobs so boundary is still constraint-like,
# but not clean circular topology.
angles: FloatArray = np.linspace(0, 2 * np.pi, 8, endpoint=False, dtype=np.float64)
ring_blob_centers: FloatArray = np.column_stack(
    [
        r0 * np.cos(angles) + np.random.normal(0.0, 0.22, len(angles)),
        r0 * np.sin(angles) + np.random.normal(0.0, 0.22, len(angles)),
    ]
).astype(np.float64)

blob_centers = np.vstack([blob_centers, ring_blob_centers]).astype(np.float64)
blob_sigmas = np.concatenate(
    [blob_sigmas, np.random.uniform(0.18, 0.38, size=len(angles))]
).astype(np.float64)
blob_weights = np.concatenate(
    [blob_weights, np.random.uniform(0.65, 1.35, size=len(angles))]
).astype(np.float64)

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
ax2d.set_title("CONTROL: Random Boundary")
ax2d.set_xlim(-3, 3)
ax2d.set_ylim(-3, 3)
ax2d.set_aspect("equal")
ax2d.set_xlabel("x")
ax2d.set_ylabel("y")

theta: FloatArray = np.linspace(0, 2 * np.pi, 600, dtype=np.float64)

# Reference ring only, faint
ax2d.plot(r0 * np.cos(theta), r0 * np.sin(theta), linewidth=0.9, alpha=0.25)
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

# Random boundary field background
grid_n = 180
gx: FloatArray = np.linspace(-3, 3, grid_n, dtype=np.float64)
gy: FloatArray = np.linspace(-3, 3, grid_n, dtype=np.float64)
GX, GY = np.meshgrid(gx, gy)
BG = np.zeros_like(GX, dtype=np.float64)

for center, sig, weight in zip(blob_centers, blob_sigmas, blob_weights):
    dx = GX - center[0]
    dy = GY - center[1]
    BG += weight * np.exp(-(dx * dx + dy * dy) / (2 * sig * sig))

BG = BG / (BG.max() + 1e-9)

ax2d.contour(GX, GY, BG, levels=[boundary_threshold], linewidths=1.0, alpha=0.55)
ax2d.imshow(
    BG,
    extent=(-3.0, 3.0, -3.0, 3.0),
    origin="lower",
    alpha=0.12,
    cmap="viridis",
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
ax3d.set_title("3D Projection: Random Boundary")
ax3d.set_xlim(-3, 3)
ax3d.set_ylim(-3, 3)
ax3d.set_zlim(-3, 3)
ax3d.set_xlabel("x′")
ax3d.set_ylabel("y′")
ax3d.set_zlabel("z′")

ring_x: FloatArray = r0 * np.cos(theta)
ring_y: FloatArray = r0 * np.sin(theta)
ring_z: FloatArray = np.zeros_like(theta)
ax3d.plot(ring_x, ring_y, ring_z, linewidth=0.8, alpha=0.25)

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
axm.set_title("CONTROL Metrics: Random Boundary")
axm.set_xlim(0, 900)
axm.set_ylim(0, 2.25)
axm.set_xlabel("frame")

line_cont, = axm.plot([], [], label="global continuity")
line_dist, = axm.plot([], [], label="boundary distortion")
line_occ, = axm.plot([], [], label="boundary occupancy")
line_w, = axm.plot([], [], label="latent phase spread |w|")
line_trap, = axm.plot([], [], label="orbit/trap fraction")

axm.legend(loc="upper right")

frames: list[int] = []
continuity_vals: list[float] = []
distortion_vals: list[float] = []
occupancy_vals: list[float] = []
w_vals: list[float] = []
trap_vals: list[float] = []

# -----------------------------
# HELPERS
# -----------------------------
def boundary_strength(x: FloatArray, y: FloatArray) -> FloatArray:
    B = np.zeros_like(x, dtype=np.float64)

    for center, sig, weight in zip(blob_centers, blob_sigmas, blob_weights):
        dx = x - center[0]
        dy = y - center[1]
        B += weight * np.exp(-(dx * dx + dy * dy) / (2 * sig * sig))

    max_b = float(np.max(B)) + 1e-9
    return (B / max_b).astype(np.float64)


def boundary_gradient(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    Approximate gradient of randomized boundary field.
    Used to define local radial-ish push/pull from random constraint.
    """
    eps = 1e-3

    bx1 = boundary_strength(x + eps, y)
    bx0 = boundary_strength(x - eps, y)
    by1 = boundary_strength(x, y + eps)
    by0 = boundary_strength(x, y - eps)

    gx_arr = (bx1 - bx0) / (2 * eps)
    gy_arr = (by1 - by0) / (2 * eps)

    norm = np.sqrt(gx_arr * gx_arr + gy_arr * gy_arr) + 1e-9
    return (gx_arr / norm).astype(np.float64), (gy_arr / norm).astype(np.float64)


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


def latent_vector_field_random_boundary(
    s: FloatArray,
    frame: int,
) -> tuple[FloatArray, FloatArray]:
    x = s[:, 0]
    y = s[:, 1]
    z = s[:, 2]
    w = s[:, 3]

    r = np.sqrt(x * x + y * y) + 1e-9
    phi = np.arctan2(y, x)

    B = boundary_strength(x, y)

    # Random-boundary local normal from gradient
    gn_x, gn_y = boundary_gradient(x, y)

    # Tangent is perpendicular to gradient
    gt_x = -gn_y
    gt_y = gn_x

    # Orthogonal x/y modes
    u = np.sin(2.8 * x + 0.040 * frame)
    v = np.sin(2.8 * y - 0.035 * frame)

    # Hidden phase mode
    q = np.sin(w + 0.025 * frame)

    # Nonlinear coupling retained
    coupling_xy = couple_gain * B * u * v
    coupling_w = B * q * (u - v)

    twist = np.sin(3.0 * phi - 0.050 * frame + 0.6 * w)

    tangential = swirl_gain * B * (1.0 + 0.55 * twist + 0.35 * coupling_xy)
    radial = radial_gain * B * np.sin(2.0 * phi + 0.040 * frame + coupling_xy)

    drift_x = orth_gain * u
    drift_y = orth_gain * v

    dx = drift_x + tangential * gt_x + radial * gn_x
    dy = drift_y + tangential * gt_y + radial * gn_y

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

    B = boundary_strength(x, y)
    gn_x, gn_y = boundary_gradient(x, y)

    gt_x = -gn_y
    gt_y = gn_x

    normal_v = np.abs(vx * gn_x + vy * gn_y)
    tangent_v = np.abs(vx * gt_x + vy * gt_y)

    trapped = (B > boundary_threshold) & (tangent_v > 1.35 * normal_v)
    return float(np.mean(trapped))


def classify_state(cont: float, dist: float, occ: float, trap: float) -> str:
    if dist > collapse_distortion_threshold and cont < 0.55:
        return "LOCAL COLLAPSE / GLOBAL CONTINUITY WEAK"
    if dist > collapse_distortion_threshold and cont >= 0.55:
        return "LOCAL COLLAPSE / GLOBAL CONTINUITY SURVIVES"
    if trap > 0.18:
        return "ORBIT / RANDOM-BOUNDARY TRAP"
    if occ < 0.07 and cont > reconstruct_continuity_threshold:
        return "POST-BOUNDARY RECONSTRUCTION"
    if occ > 0.20:
        return "RANDOM-BOUNDARY TRAVERSAL"
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

    field_vel, B = latent_vector_field_random_boundary(state4, frame)

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
    line_dist.set_data(frames, distortion_vals)
    line_occ.set_data(frames, occupancy_vals)
    line_w.set_data(frames, w_vals)
    line_trap.set_data(frames, trap_vals)

    info.set_text(
        f"frame = {frame}\n"
        f"state = {phase_label}\n"
        f"global continuity = {cont:.3f}\n"
        f"boundary distortion = {dist:.3f}\n"
        f"boundary occupancy = {occ:.3f}\n"
        f"latent |w| spread = {w_spread:.3f}\n"
        f"orbit/trap fraction = {trap:.3f}\n"
        f"CONTROL = RANDOM BOUNDARY"
    )

    return (
        sc2d,
        sc3d_holder[0],
        line_cont,
        line_dist,
        line_occ,
        line_w,
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