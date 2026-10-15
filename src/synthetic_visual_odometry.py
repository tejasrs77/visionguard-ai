
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. CONFIGURATION
# ==========================================

OUTPUT_DIR = Path("runs/phase8")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_LANDMARKS = 500
NUM_FRAMES = 30

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

FOCAL_LENGTH = 900.0

# Reduced image measurement noise
NOISE_STD = 0.10

MIN_POSE_INLIERS = 15

rng = np.random.default_rng(42)

K = np.array([
    [FOCAL_LENGTH, 0, IMAGE_WIDTH / 2],
    [0, FOCAL_LENGTH, IMAGE_HEIGHT / 2],
    [0, 0, 1]
], dtype=np.float64)


# ==========================================
# 2. GENERATE STATIC 3D LANDMARKS
# ==========================================

landmarks = np.column_stack([
    rng.uniform(-10, 10, NUM_LANDMARKS),
    rng.uniform(-5, 5, NUM_LANDMARKS),
    rng.uniform(15, 35, NUM_LANDMARKS)
])

print("\n========== SYNTHETIC ENVIRONMENT ==========")
print("Generated 3D landmarks:", len(landmarks))
print("Total camera frames:", NUM_FRAMES)
print("Image noise standard deviation:", NOISE_STD)


# ==========================================
# 3. GENERATE GROUND-TRUTH CAMERA TRAJECTORY
# ==========================================

ground_truth = []

for i in range(NUM_FRAMES):

    # Increased baseline for stronger parallax
    x = i * 0.60

    # Slight curved movement
    y = 0.4 * np.sin(i * 0.15)

    # Forward movement
    z = i * 0.15

    ground_truth.append([x, y, z])

ground_truth = np.array(
    ground_truth,
    dtype=np.float64
)

print("\nGround-truth start:", ground_truth[0])
print("Ground-truth end:", ground_truth[-1])


# ==========================================
# 4. PROJECT 3D LANDMARKS INTO IMAGE
# ==========================================

def project_points(camera_position):
    """
    Project fixed world landmarks into a
    simulated camera.

    Camera orientation is fixed.

    X_camera = X_world - camera_position
    """

    camera_points = landmarks - camera_position

    depth = camera_points[:, 2]

    safe_depth = np.maximum(depth, 1e-6)

    normalized_x = camera_points[:, 0] / safe_depth
    normalized_y = camera_points[:, 1] / safe_depth

    u = (
        FOCAL_LENGTH * normalized_x
        + IMAGE_WIDTH / 2
    )

    v = (
        FOCAL_LENGTH * normalized_y
        + IMAGE_HEIGHT / 2
    )

    valid = (
        (depth > 0.1)
        & (u >= 0)
        & (u < IMAGE_WIDTH)
        & (v >= 0)
        & (v < IMAGE_HEIGHT)
    )

    points = np.column_stack([u, v])

    points += rng.normal(
        0,
        NOISE_STD,
        points.shape
    )

    return points, valid


# ==========================================
# 5. INITIALIZE VISUAL ODOMETRY
# ==========================================

estimated_positions = [
    np.array([0.0, 0.0, 0.0])
]

# Camera-to-world rotation
R_wc = np.eye(3, dtype=np.float64)

# Camera center in world coordinates
C_w = np.zeros(3, dtype=np.float64)

successful_steps = 0
failed_steps = 0

total_matches = 0
total_pose_inliers = 0

step_results = []


# ==========================================
# 6. ESTIMATE RELATIVE CAMERA MOTION
# ==========================================

print("\n========== VISUAL ODOMETRY PROCESSING ==========")

for i in range(NUM_FRAMES - 1):

    pts1_all, valid1 = project_points(
        ground_truth[i]
    )

    pts2_all, valid2 = project_points(
        ground_truth[i + 1]
    )

    # Keep only landmarks visible in both views
    common = valid1 & valid2

    pts1 = pts1_all[common]
    pts2 = pts2_all[common]

    match_count = len(pts1)

    if match_count < 8:
        print(
            f"Frame {i:02d}->{i + 1:02d} | "
            "FAILED: insufficient correspondences"
        )

        failed_steps += 1
        estimated_positions.append(C_w.copy())
        step_results.append((i, match_count, 0, False))
        continue

    # ======================================
    # 7. ESTIMATE ESSENTIAL MATRIX
    # ======================================

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None or E.shape != (3, 3):
        print(
            f"Frame {i:02d}->{i + 1:02d} | "
            "FAILED: essential matrix estimation"
        )

        failed_steps += 1
        estimated_positions.append(C_w.copy())
        step_results.append((i, match_count, 0, False))
        continue

    # ======================================
    # 8. RECOVER RELATIVE CAMERA POSE
    # ======================================

    pose_inliers, R, t, pose_mask = cv2.recoverPose(
        E,
        pts1,
        pts2,
        K,
        mask=mask.copy()
    )

    total_matches += match_count
    total_pose_inliers += int(pose_inliers)

    if pose_inliers < MIN_POSE_INLIERS:
        print(
            f"Frame {i:02d}->{i + 1:02d} | "
            f"Matches: {match_count} | "
            f"Pose inliers: {pose_inliers} | "
            "REJECTED"
        )

        failed_steps += 1
        estimated_positions.append(C_w.copy())
        step_results.append(
            (i, match_count, int(pose_inliers), False)
        )
        continue

    # ======================================
    # 9. RESOLVE TRANSLATION SCALE
    # ======================================

    # Monocular VO cannot recover metric
    # translation scale from E alone.
    #
    # We supply ground-truth step length
    # only for this controlled experiment.

    true_step = np.linalg.norm(
        ground_truth[i + 1] - ground_truth[i]
    )

    # recoverPose convention:
    # X_camera2 = R @ X_camera1 + t
    #
    # Camera 2 center expressed in camera 1
    # coordinates:
    # -R.T @ t

    relative_center = -R.T @ t.reshape(3)

    relative_center *= true_step

    # Convert relative displacement into
    # world coordinates.
    C_w = C_w + R_wc @ relative_center

    # Update camera-to-world orientation.
    R_wc = R_wc @ R.T

    estimated_positions.append(C_w.copy())

    successful_steps += 1

    step_results.append(
        (i, match_count, int(pose_inliers), True)
    )

    print(
        f"Frame {i:02d}->{i + 1:02d} | "
        f"Matches: {match_count} | "
        f"Pose inliers: {pose_inliers} | "
        "SUCCESS"
    )


# ==========================================
# 10. EVALUATE ESTIMATED TRAJECTORY
# ==========================================

estimated_positions = np.array(
    estimated_positions,
    dtype=np.float64
)

gt_relative = ground_truth - ground_truth[0]

errors = np.linalg.norm(
    estimated_positions - gt_relative,
    axis=1
)

rmse = np.sqrt(np.mean(errors ** 2))

final_error = errors[-1]

success_rate = (
    successful_steps / (NUM_FRAMES - 1)
) * 100

mean_error = np.mean(errors)
max_error = np.max(errors)

print("\n========== VISUAL ODOMETRY RESULTS ==========")

print("Total frames:", NUM_FRAMES)

print(
    "Total motion estimation steps:",
    NUM_FRAMES - 1
)

print("Successful pose estimates:", successful_steps)
print("Failed pose estimates:", failed_steps)

print(
    "Pose estimation success rate:",
    round(success_rate, 2),
    "%"
)

print("Total correspondences:", total_matches)
print("Total pose inliers:", total_pose_inliers)

print(
    "Trajectory RMSE:",
    round(float(rmse), 4),
    "simulation units"
)

print(
    "Mean position error:",
    round(float(mean_error), 4),
    "simulation units"
)

print(
    "Maximum position error:",
    round(float(max_error), 4),
    "simulation units"
)

print(
    "Final position error:",
    round(float(final_error), 4),
    "simulation units"
)


# ==========================================
# 11. SAVE TRAJECTORY DATA
# ==========================================

trajectory_data = np.column_stack([
    np.arange(NUM_FRAMES),
    gt_relative,
    estimated_positions,
    errors
])

csv_path = OUTPUT_DIR / "vo_trajectory_data.csv"

np.savetxt(
    csv_path,
    trajectory_data,
    delimiter=",",
    header=(
        "frame,"
        "gt_x,gt_y,gt_z,"
        "estimated_x,estimated_y,estimated_z,"
        "position_error"
    ),
    comments=""
)

print("\nSaved trajectory data:", csv_path)


# ==========================================
# 12. PLOT GROUND TRUTH VS ESTIMATED PATH
# ==========================================

plt.figure(figsize=(10, 7))

plt.plot(
    gt_relative[:, 0],
    gt_relative[:, 2],
    label="Ground Truth",
    linewidth=2,
    marker="o",
    markersize=3
)

plt.plot(
    estimated_positions[:, 0],
    estimated_positions[:, 2],
    label="Estimated Visual Odometry",
    linewidth=2,
    linestyle="--",
    marker="x",
    markersize=4
)

plt.xlabel("X Position (simulation units)")
plt.ylabel("Z Position (simulation units)")

plt.title(
    "VisionGuard-AI: Synthetic Visual Odometry"
)

plt.legend()
plt.grid(True)
plt.axis("equal")

trajectory_path = OUTPUT_DIR / "vo_trajectory.png"

plt.savefig(
    trajectory_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved trajectory plot:", trajectory_path)


# ==========================================
# 13. PLOT POSITION ERROR
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    np.arange(NUM_FRAMES),
    errors,
    linewidth=2,
    marker="o",
    markersize=3
)

plt.xlabel("Frame Index")
plt.ylabel("Position Error (simulation units)")

plt.title(
    "VisionGuard-AI: Visual Odometry Position Error"
)

plt.grid(True)

error_plot_path = OUTPUT_DIR / "vo_position_error.png"

plt.savefig(
    error_plot_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved error plot:", error_plot_path)

print("\nPhase 8 - Step 4 completed!")
