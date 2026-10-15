
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. CONFIGURATION
# ==========================================

OUTPUT_DIR = Path("runs/phase8")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 100
DT = 1.0

MEASUREMENT_NOISE_STD = 3.0
ACCELERATION_STD = 0.15

rng = np.random.default_rng(42)


# ==========================================
# 2. GENERATE GROUND-TRUTH MOTION
# ==========================================

# Constant velocity in 2D
initial_position = np.array([0.0, 0.0])
true_velocity = np.array([1.5, 0.8])

time_steps = np.arange(NUM_FRAMES)

ground_truth = (
    initial_position
    + time_steps[:, None] * true_velocity
)

# Simulated noisy position observations
measurements = ground_truth + rng.normal(
    0,
    MEASUREMENT_NOISE_STD,
    ground_truth.shape
)

print("\n========== SYNTHETIC TRACKING ==========")
print("Total frames:", NUM_FRAMES)
print("True velocity:", true_velocity)
print("Measurement noise:", MEASUREMENT_NOISE_STD)


# ==========================================
# 3. INITIALIZE KALMAN FILTER
# ==========================================

# 4 state variables: x, y, vx, vy
# 2 measurements: x, y

kf = cv2.KalmanFilter(4, 2)

kf.transitionMatrix = np.array([
    [1, 0, DT, 0],
    [0, 1, 0, DT],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

kf.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)


# ==========================================
# 4. PROCESS AND MEASUREMENT NOISE
# ==========================================

# White acceleration noise model
dt2 = DT ** 2
dt3 = DT ** 3
dt4 = DT ** 4

q = ACCELERATION_STD ** 2

kf.processNoiseCov = (
    q * np.array([
        [dt4 / 4, 0, dt3 / 2, 0],
        [0, dt4 / 4, 0, dt3 / 2],
        [dt3 / 2, 0, dt2, 0],
        [0, dt3 / 2, 0, dt2]
    ], dtype=np.float32)
)

kf.measurementNoiseCov = (
    np.eye(2, dtype=np.float32)
    * MEASUREMENT_NOISE_STD ** 2
)

kf.errorCovPost = (
    np.eye(4, dtype=np.float32) * 100.0
)


# ==========================================
# 5. INITIAL STATE ESTIMATE
# ==========================================

kf.statePost = np.array([
    [measurements[0, 0]],
    [measurements[0, 1]],
    [0.0],
    [0.0]
], dtype=np.float32)

estimated_positions = [
    measurements[0].copy()
]

estimated_velocities = [
    np.array([0.0, 0.0])
]


# ==========================================
# 6. KALMAN PREDICTION AND CORRECTION
# ==========================================

print("\n========== KALMAN FILTER PROCESSING ==========")

for i in range(1, NUM_FRAMES):

    # Predict next position and velocity
    prediction = kf.predict()

    # Prepare noisy position measurement
    measurement = np.array([
        [measurements[i, 0]],
        [measurements[i, 1]]
    ], dtype=np.float32)

    # Correct predicted state
    corrected = kf.correct(measurement)

    position = corrected[:2, 0].copy()
    velocity = corrected[2:, 0].copy()

    estimated_positions.append(position)
    estimated_velocities.append(velocity)

    if i % 10 == 0:
        print(
            f"Frame {i:03d} | "
            f"Measured: {np.round(measurements[i], 2)} | "
            f"Estimated: {np.round(position, 2)}"
        )


estimated_positions = np.array(
    estimated_positions,
    dtype=np.float64
)

estimated_velocities = np.array(
    estimated_velocities,
    dtype=np.float64
)


# ==========================================
# 7. EVALUATE ACCURACY
# ==========================================

measurement_errors = np.linalg.norm(
    measurements - ground_truth,
    axis=1
)

kalman_errors = np.linalg.norm(
    estimated_positions - ground_truth,
    axis=1
)

measurement_rmse = np.sqrt(
    np.mean(measurement_errors ** 2)
)

kalman_rmse = np.sqrt(
    np.mean(kalman_errors ** 2)
)

improvement = (
    (measurement_rmse - kalman_rmse)
    / measurement_rmse
) * 100

print("\n========== KALMAN FILTER RESULTS ==========")

print(
    "Measurement RMSE:",
    round(float(measurement_rmse), 4)
)

print(
    "Kalman Filter RMSE:",
    round(float(kalman_rmse), 4)
)

print(
    "RMSE improvement:",
    round(float(improvement), 2),
    "%"
)

print(
    "Final estimated velocity:",
    np.round(estimated_velocities[-1], 3)
)

print(
    "True velocity:",
    true_velocity
)


# ==========================================
# 8. SAVE RESULTS AS CSV
# ==========================================

results = np.column_stack([
    time_steps,
    ground_truth,
    measurements,
    estimated_positions,
    measurement_errors,
    kalman_errors
])

csv_path = OUTPUT_DIR / "kalman_results.csv"

np.savetxt(
    csv_path,
    results,
    delimiter=",",
    header=(
        "frame,"
        "true_x,true_y,"
        "measured_x,measured_y,"
        "estimated_x,estimated_y,"
        "measurement_error,kalman_error"
    ),
    comments=""
)

print("\nSaved CSV:", csv_path)


# ==========================================
# 9. PLOT TRAJECTORIES
# ==========================================

plt.figure(figsize=(11, 7))

plt.plot(
    ground_truth[:, 0],
    ground_truth[:, 1],
    label="Ground Truth",
    linewidth=2.5
)

plt.scatter(
    measurements[:, 0],
    measurements[:, 1],
    label="Noisy Measurements",
    alpha=0.4,
    s=20
)

plt.plot(
    estimated_positions[:, 0],
    estimated_positions[:, 1],
    label="Kalman Estimate",
    linewidth=2
)

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.title(
    "VisionGuard-AI: Kalman State Estimation"
)

plt.legend()
plt.grid(True)
plt.axis("equal")

trajectory_path = (
    OUTPUT_DIR / "kalman_trajectory.png"
)

plt.savefig(
    trajectory_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 10. PLOT POSITION ERRORS
# ==========================================

plt.figure(figsize=(11, 5))

plt.plot(
    time_steps,
    measurement_errors,
    label="Measurement Error",
    alpha=0.7
)

plt.plot(
    time_steps,
    kalman_errors,
    label="Kalman Filter Error",
    linewidth=2
)

plt.xlabel("Frame")
plt.ylabel("Euclidean Position Error")

plt.title(
    "VisionGuard-AI: Measurement vs Kalman Error"
)

plt.legend()
plt.grid(True)

error_path = (
    OUTPUT_DIR / "kalman_error_comparison.png"
)

plt.savefig(
    error_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved trajectory plot:", trajectory_path)
print("Saved error comparison:", error_path)

print("\nPhase 8 - Step 5 completed!")
