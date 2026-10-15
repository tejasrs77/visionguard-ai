
from pathlib import Path

import cv2
import numpy as np


# ==========================================
# 1. CONFIGURATION
# ==========================================

VIDEO_PATH = Path("data/videos/test.mp4")

OUTPUT_DIR = Path("runs/phase8")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_1_INDEX = 100
FRAME_2_INDEX = 105

TARGET_WIDTH = 960


# ==========================================
# 2. READ VIDEO FRAMES
# ==========================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open video.")


def read_frame(index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    success, frame = cap.read()

    if not success:
        raise RuntimeError(
            f"Could not read frame {index}"
        )

    h, w = frame.shape[:2]
    new_h = round(h * TARGET_WIDTH / w)

    return cv2.resize(
        frame,
        (TARGET_WIDTH, new_h)
    )


try:
    frame1 = read_frame(FRAME_1_INDEX)
    frame2 = read_frame(FRAME_2_INDEX)

finally:
    cap.release()


# ==========================================
# 3. CAMERA INTRINSIC MATRIX
# ==========================================

height, width = frame1.shape[:2]

# Approximate intrinsics for demonstration
focal = float(max(width, height))

K = np.array([
    [focal, 0, width / 2],
    [0, focal, height / 2],
    [0, 0, 1]
], dtype=np.float64)


# ==========================================
# 4. ORB FEATURE MATCHING
# ==========================================

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=2000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    raise RuntimeError("ORB descriptors unavailable.")

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

knn_matches = matcher.knnMatch(
    des1,
    des2,
    k=2
)

good_matches = []

for pair in knn_matches:
    if len(pair) < 2:
        continue

    best, second = pair

    if best.distance < 0.75 * second.distance:
        good_matches.append(best)

if len(good_matches) < 8:
    raise RuntimeError("Insufficient feature matches.")

pts1 = np.float64([
    kp1[m.queryIdx].pt for m in good_matches
])

pts2 = np.float64([
    kp2[m.trainIdx].pt for m in good_matches
])

print("\nGood feature matches:", len(good_matches))


# ==========================================
# 5. MATCH DISPLACEMENT
# ==========================================

displacements = np.linalg.norm(
    pts2 - pts1,
    axis=1
)

print(
    "Median feature displacement:",
    round(float(np.median(displacements)), 3),
    "pixels"
)


# ==========================================
# 6. ESSENTIAL MATRIX + RANSAC
# ==========================================

E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None:
    raise RuntimeError(
        "Essential matrix estimation failed."
    )

if E.shape != (3, 3):
    raise RuntimeError(
        f"Unexpected essential matrix shape: {E.shape}"
    )

inlier_count = int(np.count_nonzero(mask))
inlier_ratio = inlier_count / len(good_matches)

print("\nRANSAC inliers:", inlier_count)
print(
    "RANSAC inlier ratio:",
    round(inlier_ratio * 100, 2),
    "%"
)


# ==========================================
# 7. RECOVER RELATIVE POSE
# ==========================================

pose_inliers, R, t, pose_mask = cv2.recoverPose(
    E,
    pts1,
    pts2,
    K,
    mask=mask.copy()
)

print("\nRotation matrix:")
print(R)

print("\nTranslation direction:")
print(t)

print("\nPose inliers:", pose_inliers)


# ==========================================
# 8. ESTIMATE ROTATION ANGLE
# ==========================================

trace_value = np.trace(R)

cos_angle = (trace_value - 1) / 2
cos_angle = np.clip(cos_angle, -1.0, 1.0)

rotation_angle = np.degrees(
    np.arccos(cos_angle)
)

print(
    "\nEstimated rotation angle:",
    round(float(rotation_angle), 3),
    "degrees"
)


# ==========================================
# 9. VISUALIZE GEOMETRIC INLIERS
# ==========================================

inlier_matches = [
    m
    for m, valid in zip(
        good_matches,
        pose_mask.ravel()
    )
    if valid != 0
]

visualization = cv2.drawMatches(
    frame1,
    kp1,
    frame2,
    kp2,
    inlier_matches[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

output_path = (
    OUTPUT_DIR / "relative_motion_matches.jpg"
)

cv2.imwrite(str(output_path), visualization)

print("\nSaved visualization:", output_path)

print("\nPhase 8 - Step 2 completed!")
