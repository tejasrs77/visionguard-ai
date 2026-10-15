
from pathlib import Path

import cv2
import numpy as np


# ==========================================
# 1. FILE PATHS
# ==========================================

IMAGE_1 = Path("data/sfm/view1.png")
IMAGE_2 = Path("data/sfm/view2.png")

OUTPUT_DIR = Path("runs/phase6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD IMAGES
# ==========================================

image1 = cv2.imread(str(IMAGE_1))
image2 = cv2.imread(str(IMAGE_2))

if image1 is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_1}")

if image2 is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_2}")

print("\nOriginal image dimensions:")
print("Image 1:", image1.shape)
print("Image 2:", image2.shape)


# ==========================================
# 3. MATCH IMAGE RESOLUTIONS
# ==========================================

height, width = image1.shape[:2]

if image2.shape[:2] != image1.shape[:2]:
    image2 = cv2.resize(
        image2,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

print("\nFinal image dimensions:")
print("Image 1:", image1.shape)
print("Image 2:", image2.shape)


# ==========================================
# 4. GRAYSCALE CONVERSION
# ==========================================

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# ==========================================
# 5. APPROXIMATE CAMERA MATRIX
# ==========================================

# Demonstration-only intrinsics
focal = float(max(width, height))

K = np.array([
    [focal, 0, width / 2],
    [0, focal, height / 2],
    [0, 0, 1]
], dtype=np.float64)

print("\nCamera intrinsic matrix:")
print(K)


# ==========================================
# 6. ORB FEATURE DETECTION
# ==========================================

orb = cv2.ORB_create(nfeatures=3000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

print("\nORB Feature Detection:")
print("Image 1 keypoints:", len(kp1))
print("Image 2 keypoints:", len(kp2))

if des1 is None or des2 is None:
    raise RuntimeError(
        "ORB could not detect enough features."
    )


# ==========================================
# 7. FEATURE MATCHING
# ==========================================

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

    first, second = pair

    if first.distance < 0.75 * second.distance:
        good_matches.append(first)

print("\nFeature Matching:")
print("Good matches:", len(good_matches))

if len(good_matches) < 8:
    raise RuntimeError(
        "Not enough feature matches. "
        "Try two images with more overlapping details."
    )


# ==========================================
# 8. EXTRACT MATCHED COORDINATES
# ==========================================

points1 = np.float64([
    kp1[match.queryIdx].pt
    for match in good_matches
])

points2 = np.float64([
    kp2[match.trainIdx].pt
    for match in good_matches
])


# ==========================================
# 9. ESSENTIAL MATRIX WITH RANSAC
# ==========================================

E, mask = cv2.findEssentialMat(
    points1,
    points2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None:
    raise RuntimeError(
        "Essential matrix estimation failed."
    )

# OpenCV can return multiple candidate matrices
if E.shape != (3, 3):
    if E.shape[1] == 3 and E.shape[0] % 3 == 0:
        print(
            "Multiple essential matrix candidates "
            "returned; using the first."
        )
        E = E[:3, :]
    else:
        raise RuntimeError(
            f"Unexpected essential matrix shape: {E.shape}"
        )

ransac_inliers = int(np.count_nonzero(mask))

print("\nEssential Matrix:")
print(E)

print("\nRANSAC inliers:", ransac_inliers)
print("Total good matches:", len(good_matches))


# ==========================================
# 10. RECOVER CAMERA POSE
# ==========================================

pose_inliers, R, t, pose_mask = cv2.recoverPose(
    E,
    points1,
    points2,
    K,
    mask=mask.copy()
)

print("\nRotation Matrix:")
print(R)

print("\nTranslation Direction:")
print(t)

print("\nPose inliers:", pose_inliers)


# ==========================================
# 11. VISUALIZE INLIER MATCHES
# ==========================================

inlier_matches = [
    match
    for match, valid in zip(
        good_matches,
        pose_mask.ravel()
    )
    if valid != 0
]

print("Final geometrically verified matches:",
      len(inlier_matches))

visualization = cv2.drawMatches(
    image1,
    kp1,
    image2,
    kp2,
    inlier_matches[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

output_path = OUTPUT_DIR / "relative_pose_matches.jpg"

if not cv2.imwrite(str(output_path), visualization):
    raise RuntimeError("Failed to save visualization")

print("\nSaved visualization:", output_path)


# ==========================================
# 12. DISPLAY RESULTS
# ==========================================

display_height, display_width = visualization.shape[:2]

scale = min(
    1400 / display_width,
    800 / display_height,
    1.0
)

preview = cv2.resize(
    visualization,
    (
        max(1, int(display_width * scale)),
        max(1, int(display_height * scale))
    )
)

cv2.imshow(
    "VisionGuard AI - Relative Camera Pose",
    preview
)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nPhase 6 - Step 4 completed!")
