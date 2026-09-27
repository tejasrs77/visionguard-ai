import cv2


image_path = "data/images/test.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image could not be loaded.")
else:
    print("Image loaded successfully!")

    print("Shape:", image.shape)
    print("Height:", image.shape[0])
    print("Width:", image.shape[1])
    print("Channels:", image.shape[2])

    x = 200
    y = 100

    pixel = image[y, x]

    print("Pixel at (200, 100):", pixel)
    print("Blue:", pixel[0])
    print("Green:", pixel[1])
    print("Red:", pixel[2])

    resized = cv2.resize(image, (640, 640))

    x1 = 100
    y1 = 100
    x2 = 600
    y2 = 700

    cropped = image[y1:y2, x1:x2]

    cv2.imshow("Cropped Image", cropped)

    print("Original shape:", image.shape)
    print("Resized shape:", resized.shape)

    cv2.imshow("Original Image", image)
    cv2.imshow("Resized Image", resized)
    cv2.imshow("Cropped Image", cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()