
import numpy as np

# Example camera intrinsic parameters
fx = 800
fy = 800
cx = 640
cy = 360

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)

print("Camera Intrinsic Matrix:")
print(K)

# Example 3D point in camera coordinates
# X, Y, Z measured in meters
point_3d = np.array([1.0, 0.5, 5.0])

X, Y, Z = point_3d

if Z <= 0:
    raise ValueError("Point must be in front of camera")

# Project 3D point to image coordinates
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy

print("\n3D Point:", point_3d)
print("Projected 2D Pixel:", (u, v))
