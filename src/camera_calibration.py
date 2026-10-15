
from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = Path("data/calibration")
OUTPUT_DIR = Path("runs/phase6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of INNER corners: columns, rows
PATTERN_SIZE = (9, 6)
SQUARE_SIZE = 1.0  # Arbitrary units

# Known 3D positions of chessboard corners
object_points_template = np.zeros(
    (PATTERN_SIZE[0] * PATTERN_SIZE[1], 3),
    dtype=np.float32
)

object_points_template[:, :2] = (
    np.mgrid[
        0:PATTERN_SIZE[0],
        0:PATTERN_SIZE[1]
    ].T.reshape(-1, 2)
)

object_points_template *= SQUARE_SIZE

object_points = []
image_points = []

image_size = None

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
) if IMAGE_DIR.exists() else []

if not image_paths:
    raise RuntimeError(
        "No calibration images found in data/calibration"
    )

for image_path in image_paths:
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    current_size = (gray.shape[1], gray.shape[0])

    if image_size is None:
        image_size = current_size
    elif current_size != image_size:
        print("Skipping different image size:", image_path)
        continue

    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE
    )

    if not found:
        print("Corners NOT found:", image_path.name)
        continue

    refined_corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
    )

    object_points.append(object_points_template.copy())
    image_points.append(refined_corners)

    print("Corners found:", image_path.name)

if len(object_points) < 10:
    raise RuntimeError(
        f"Only {len(object_points)} valid images. "
        "Collect at least 10 different chessboard views."
    )

# Estimate camera intrinsics and distortion
rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)

print("\nCalibration successful!")
print("Valid images:", len(object_points))
print("Reprojection RMS error:", rms)
print("\nCamera intrinsic matrix:")
print(camera_matrix)
print("\nDistortion coefficients:")
print(dist_coeffs)

# Save calibration for future scripts
np.savez(
    OUTPUT_DIR / "camera_calibration.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_size=np.array(image_size)
)

# Undistort one sample image
sample = cv2.imread(str(image_paths[0]))

if sample is not None:
    undistorted = cv2.undistort(
        sample,
        camera_matrix,
        dist_coeffs
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "undistorted.jpg"),
        undistorted
    )

print("\nSaved calibration to:", OUTPUT_DIR)
