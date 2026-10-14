
# VisionGuard-AI
# Phase 2 - Intersection over Union (IoU)

def calculate_iou(box1, box2):

    # STEP 1: Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # STEP 2: Calculate intersection area
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    # STEP 3: Calculate individual box areas
    box1_area = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    box2_area = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    # STEP 4: Calculate union area
    union_area = (
        box1_area + box2_area - intersection_area
    )

    # STEP 5: Calculate IoU
    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area

    return iou


# Example bounding boxes in xyxy format
ground_truth = [100, 100, 300, 300]
predicted = [150, 150, 350, 350]

iou_score = calculate_iou(
    ground_truth,
    predicted
)

print("Ground Truth Box:", ground_truth)
print("Predicted Box:", predicted)
print("IoU Score:", round(iou_score, 4))

if iou_score >= 0.5:
    print("Result: IoU >= 0.5")
else:
    print("Result: IoU < 0.5")
