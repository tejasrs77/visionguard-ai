
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path("data/calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# 10 columns x 7 rows of squares
COLS, ROWS = 10, 7
SQUARE = 80

board = np.full(
    (ROWS * SQUARE, COLS * SQUARE),
    255,
    dtype=np.uint8
)

for row in range(ROWS):
    for col in range(COLS):
        if (row + col) % 2 == 0:
            cv2.rectangle(
                board,
                (col * SQUARE, row * SQUARE),
                ((col + 1) * SQUARE,
                 (row + 1) * SQUARE),
                0,
                -1
            )

# Simulated pinhole camera
width, height = 1280, 720
K = np.array([
    [900, 0, width / 2],
    [0, 900, height / 2],
    [0, 0, 1]
], dtype=np.float64)

# Board coordinates in arbitrary units
world_corners = np.array([
    [0, 0, 0],
    [10, 0, 0],
    [10, 7, 0],
    [0, 7, 0]
], dtype=np.float32)

source_corners = np.array([
    [0, 0],
    [board.shape[1] - 1, 0],
    [board.shape[1] - 1, board.shape[0] - 1],
    [0, board.shape[0] - 1]
], dtype=np.float32)

for i in range(20):
    # Vary camera viewpoint and board position
    angles = np.deg2rad(rng.uniform(-22, 22, 3))
    rotation_vector = angles.astype(np.float64)

    translation = np.array([
        rng.uniform(-5.5, -4.5),
        rng.uniform(-3.8, -3.2),
        rng.uniform(14, 21)
    ], dtype=np.float64)

    projected, _ = cv2.projectPoints(
        world_corners,
        rotation_vector,
        translation,
        K,
        np.zeros(5)
    )

    destination = projected.reshape(-1, 2).astype(np.float32)

    H = cv2.getPerspectiveTransform(
        source_corners,
        destination
    )

    image = cv2.warpPerspective(
        board,
        H,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=220
    )

    image = cv2.GaussianBlur(image, (3, 3), 0.5)

    filename = OUTPUT_DIR / f"image{i+1:02d}.jpg"
    cv2.imwrite(str(filename), image)

    print("Generated:", filename)

print("\nDataset generation complete!")
print("Total images: 20")
print("Ground-truth camera matrix:")
print(K)
