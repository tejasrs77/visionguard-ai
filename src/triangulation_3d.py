
from pathlib import Path

import cv2
import numpy as np


# =====================================
# 1. CONFIGURATION
# =====================================

IMAGE_1 = Path("data/sfm/view1.png")
IMAGE_2 = Path("data/sfm/view2.png")

OUTPUT_DIR = Path("runs/phase6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. LOAD IMAGES
# =====================================

img1 = cv2.imread(str(IMAGE_1))
img2 = cv2.imread(str(IMAGE_2))

if img1 is None or img2 is None:
    raise FileNotFoundError(
        "Could not load the two SfM images."
    )

height, width = img1.shape[:2]

if img2.shape[:2] != img1.shape[:2]:
    img2 = cv2.resize(img2, (width, height))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# =====================================
# 3. APPROXIMATE CAMERA INTRINSICS
# =====================================

focal = float(max(width, height))

K = np.array([
    [focal, 0, width / 2],
    [0, focal, height / 2],
    [0, 0, 1]
], dtype=np.float64)


# =====================================
# 4. ORB FEATURE MATCHING
# =====================================

orb = cv2.ORB_create(nfeatures=3000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    raise RuntimeError("No descriptors detected.")

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

knn_matches = matcher.knnMatch(des1, des2, k=2)

good_matches = []

for pair in knn_matches:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

if len(good_matches) < 8:
    raise RuntimeError("Not enough feature matches.")

pts1 = np.float64([
    kp1[m.queryIdx].pt for m in good_matches
])

pts2 = np.float64([
    kp2[m.trainIdx].pt for m in good_matches
])

print("Good matches:", len(good_matches))


# =====================================
# 5. ESSENTIAL MATRIX
# =====================================

E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None or E.shape != (3, 3):
    raise RuntimeError(
        "Essential matrix estimation failed."
    )


# =====================================
# 6. RECOVER CAMERA POSE
# =====================================

pose_inliers, R, t, pose_mask = cv2.recoverPose(
    E,
    pts1,
    pts2,
    K,
    mask=mask.copy()
)

valid = pose_mask.ravel() != 0

pts1_inliers = pts1[valid]
pts2_inliers = pts2[valid]

print("Pose inliers:", pose_inliers)

if len(pts1_inliers) < 8:
    raise RuntimeError(
        "Not enough geometrically verified points."
    )


# =====================================
# 7. CAMERA PROJECTION MATRICES
# =====================================

P1 = K @ np.hstack([
    np.eye(3),
    np.zeros((3, 1))
])

P2 = K @ np.hstack([R, t])


# =====================================
# 8. TRIANGULATE 3D POINTS
# =====================================

points_4d = cv2.triangulatePoints(
    P1,
    P2,
    pts1_inliers.T,
    pts2_inliers.T
)

# Convert homogeneous coordinates to 3D
w = points_4d[3]

valid_w = np.abs(w) > 1e-10

points_3d = (
    points_4d[:3, valid_w] /
    w[valid_w]
).T

# Keep finite points in front of both cameras
finite = np.isfinite(points_3d).all(axis=1)

points_3d = points_3d[finite]

depth1 = points_3d[:, 2]

points_camera2 = (
    R @ points_3d.T + t
).T

depth2 = points_camera2[:, 2]

positive_depth = (depth1 > 0) & (depth2 > 0)

points_3d = points_3d[positive_depth]

print("Triangulated 3D points:", len(points_3d))

if len(points_3d) == 0:
    raise RuntimeError(
        "No valid 3D points reconstructed."
    )


# =====================================
# 9. SAVE 3D POINT CLOUD
# =====================================

output_path = OUTPUT_DIR / "triangulated_points.xyz"

np.savetxt(
    output_path,
    points_3d,
    fmt="%.6f"
)

print("\nFirst five 3D points:")
print(points_3d[:5])

print("\nSaved point cloud:", output_path)
print("Note: Reconstructed scale is arbitrary.")
print("\nPhase 6 - Step 5 completed!")
