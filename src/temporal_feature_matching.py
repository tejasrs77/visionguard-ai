
from pathlib import Path

import cv2


# ==========================================
# 1. CONFIGURATION
# ==========================================

VIDEO_PATH = Path("data/videos/test.mp4")

OUTPUT_DIR = Path("runs/phase8")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "temporal_matches.jpg"

FRAME_1_INDEX = 100
FRAME_2_INDEX = 105

TARGET_WIDTH = 960


# ==========================================
# 2. READ SELECTED VIDEO FRAMES
# ==========================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


def read_frame(frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = cap.read()

    if not success:
        raise RuntimeError(
            f"Could not read frame {frame_index}"
        )

    height, width = frame.shape[:2]

    new_height = int(height * TARGET_WIDTH / width)

    frame = cv2.resize(
        frame,
        (TARGET_WIDTH, new_height)
    )

    return frame


try:
    frame1 = read_frame(FRAME_1_INDEX)
    frame2 = read_frame(FRAME_2_INDEX)

finally:
    cap.release()


# ==========================================
# 3. GRAYSCALE CONVERSION
# ==========================================

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


# ==========================================
# 4. ORB FEATURE DETECTION
# ==========================================

orb = cv2.ORB_create(nfeatures=2000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

print("\nORB Feature Detection")
print("Frame 1 keypoints:", len(kp1))
print("Frame 2 keypoints:", len(kp2))

if des1 is None or des2 is None:
    raise RuntimeError(
        "ORB could not compute descriptors."
    )


# ==========================================
# 5. FEATURE MATCHING
# ==========================================

matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

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


# ==========================================
# 6. DISPLAY MATCH STATISTICS
# ==========================================

print("\nTemporal Feature Matching")
print("First frame index:", FRAME_1_INDEX)
print("Second frame index:", FRAME_2_INDEX)
print("Good matches:", len(good_matches))

print(
    "Time difference:",
    round(
        (FRAME_2_INDEX - FRAME_1_INDEX) / 29.97,
        3
    ),
    "seconds"
)


# ==========================================
# 7. VISUALIZE MATCHES
# ==========================================

good_matches = sorted(
    good_matches,
    key=lambda match: match.distance
)

visualization = cv2.drawMatches(
    frame1,
    kp1,
    frame2,
    kp2,
    good_matches[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

if not cv2.imwrite(str(OUTPUT_PATH), visualization):
    raise RuntimeError(
        "Failed to save feature visualization."
    )

print("\nSaved visualization:", OUTPUT_PATH)

cv2.imshow(
    "VisionGuard AI - Temporal Feature Matching",
    visualization
)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nPhase 8 - Step 1 completed!")
