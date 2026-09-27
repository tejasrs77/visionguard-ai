import cv2


image = cv2.imread("data/images/test.png")

if image is None:
    raise FileNotFoundError("Image could not be loaded.")


# Bounding box coordinates
x1 = 400
y1 = 70

x2 = 700
y2 = 800


# Draw bounding box
cv2.rectangle(
    image,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3
)


# Add object label
cv2.putText(
    image,
    "Person",
    (x1, y1 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)


cv2.imshow("Bounding Box", image)

cv2.waitKey(0)
cv2.destroyAllWindows()