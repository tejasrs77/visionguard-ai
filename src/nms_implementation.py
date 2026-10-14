
# VisionGuard-AI
# Phase 2 - Non-Maximum Suppression

from iou_calculation import calculate_iou


def non_max_suppression(boxes, scores, iou_threshold=0.5):

    # Combine each bounding box with its confidence score
    detections = list(zip(boxes, scores))

    # Sort detections by confidence (highest first)
    detections.sort(
        key=lambda detection: detection[1],
        reverse=True
    )

    kept_boxes = []

    while detections:

        # Keep the highest-confidence detection
        best_box, best_score = detections.pop(0)

        kept_boxes.append((best_box, best_score))

        remaining = []

        for box, score in detections:

            # Compare remaining box with selected box
            iou = calculate_iou(best_box, box)

            # Keep only boxes below or equal to threshold
            if iou <= iou_threshold:
                remaining.append((box, score))

        detections = remaining

    return kept_boxes


# Example: Three overlapping predictions of one car
boxes = [
    [100, 100, 300, 300],
    [110, 110, 310, 310],
    [120, 120, 320, 320]
]

scores = [0.95, 0.88, 0.72]

print("Before NMS:", len(boxes), "boxes")

results = non_max_suppression(
    boxes,
    scores,
    iou_threshold=0.5
)

print("After NMS:", len(results), "boxes")

for box, score in results:
    print("Box:", box)
    print("Confidence:", score)
