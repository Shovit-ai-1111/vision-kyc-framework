import unicodedata
import numpy as np


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y))
            )
        previous = current
    return previous[-1]


def ocr_counts(truth, prediction):
    # NFC, whitespace collapse only. Preserve case, punctuation and Nepali digits.
    truth = " ".join(unicodedata.normalize("NFC", truth).split())
    prediction = " ".join(unicodedata.normalize("NFC", prediction).split())
    return {
        "character_errors": edit_distance(truth, prediction),
        "characters": len(truth),
        "word_errors": edit_distance(truth.split(), prediction.split()),
        "words": len(truth.split()),
    }


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def binary_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int((labels & predictions).sum())
    fn = int((labels & ~predictions).sum())
    fp = int((~labels & predictions).sum())
    tn = int((~labels & ~predictions).sum())
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": float((labels == predictions).mean()) if len(labels) else None,
        "far": fp / (fp + tn) if fp + tn else None,
        "frr": fn / (fn + tp) if fn + tp else None,
    }


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x * y
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0
