
from pathlib import Path

import cv2
import numpy as np


VIDEO_PATH = Path("data/videos/test.mp4")
FRAME_PAIRS = [(100, 105), (100, 115), (100, 130), (100, 160)]
TARGET_WIDTH = 960


def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, frame = cap.read()

    if not success:
        raise RuntimeError(f"Cannot read frame {index}")

    h, w = frame.shape[:2]
    new_h = round(h * TARGET_WIDTH / w)

    return cv2.resize(frame, (TARGET_WIDTH, new_h))


def analyze_pair(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape

    focal = float(max(w, h))

    K = np.array([
        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    orb = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des1, des2, k=2)

    good = []

    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 8:
        return None

    pts1 = np.float64([
        kp1[m.queryIdx].pt for m in good
    ])

    pts2 = np.float64([
        kp2[m.trainIdx].pt for m in good
    ])

    displacement = np.linalg.norm(
        pts2 - pts1,
        axis=1
    )

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None or E.shape != (3, 3):
        return None

    pose_inliers, R, t, pose_mask = cv2.recoverPose(
        E, pts1, pts2, K, mask=mask.copy()
    )

    cosine = np.clip(
        (np.trace(R) - 1) / 2,
        -1.0,
        1.0
    )

    angle = np.degrees(np.arccos(cosine))

    return {
        "matches": len(good),
        "ransac": int(np.count_nonzero(mask)),
        "pose_inliers": int(pose_inliers),
        "displacement": float(np.median(displacement)),
        "rotation": float(angle)
    }


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    try:
        for start, end in FRAME_PAIRS:
            frame1 = read_frame(cap, start)
            frame2 = read_frame(cap, end)

            result = analyze_pair(frame1, frame2)

            print(f"\nFrames {start} -> {end}")

            if result is None:
                print("Insufficient geometric information")
                continue

            print("Good matches:", result["matches"])
            print("RANSAC inliers:", result["ransac"])
            print("Pose inliers:", result["pose_inliers"])
            print(
                "Median displacement:",
                round(result["displacement"], 2),
                "pixels"
            )
            print(
                "Rotation angle:",
                round(result["rotation"], 3),
                "degrees"
            )

    finally:
        cap.release()

    print("\nMotion diagnostics completed!")


if __name__ == "__main__":
    main()
