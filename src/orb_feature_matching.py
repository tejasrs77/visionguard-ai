
from pathlib import Path

import cv2

IMAGE_1 = Path("data/calibration/image01.jpg")
IMAGE_2 = Path("data/calibration/image05.jpg")

OUTPUT_DIR = Path("runs/phase6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image1 = cv2.imread(str(IMAGE_1))
image2 = cv2.imread(str(IMAGE_2))

if image1 is None or image2 is None:
    raise FileNotFoundError(
        "Could not load calibration images."
    )

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Create ORB feature detector
orb = cv2.ORB_create(nfeatures=2000)

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

print("Image 1 keypoints:", len(kp1))
print("Image 2 keypoints:", len(kp2))

if des1 is None or des2 is None:
    raise RuntimeError("No ORB descriptors detected.")

# Match binary descriptors using Hamming distance
matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)

matches = matcher.match(des1, des2)

# Sort by descriptor distance
matches = sorted(
    matches,
    key=lambda match: match.distance
)

print("Total matches:", len(matches))

# Draw 50 strongest matches
best_matches = matches[:50]

result = cv2.drawMatches(
    image1,
    kp1,
    image2,
    kp2,
    best_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

output_path = OUTPUT_DIR / "orb_matches.jpg"

cv2.imwrite(str(output_path), result)

# Resize for screen display
height, width = result.shape[:2]
scale = min(1400 / width, 800 / height, 1.0)

preview = cv2.resize(
    result,
    (
        int(width * scale),
        int(height * scale)
    )
)

cv2.imshow("VisionGuard AI - ORB Matches", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Saved matches to:", output_path)
