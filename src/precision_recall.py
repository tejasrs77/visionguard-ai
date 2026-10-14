
# VisionGuard-AI
# Phase 2 - Precision, Recall and F1-Score


def calculate_metrics(tp, fp, fn):

    # Calculate Precision
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    # Calculate Recall
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # Calculate F1-Score
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


# Example detection counts
true_positives = 8
false_positives = 1
false_negatives = 2

precision, recall, f1 = calculate_metrics(
    true_positives,
    false_positives,
    false_negatives
)

print("VisionGuard-AI Detection Evaluation")
print("-----------------------------------")

print("True Positives:", true_positives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("-----------------------------------")
