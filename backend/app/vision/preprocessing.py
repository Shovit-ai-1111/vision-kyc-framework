"""Bounded document geometry and correlated OCR variants; originals are retained."""

import cv2
import numpy as np
from app.nlp.nepali import header_type, label_fields
from app.vision.image_io import resize_max


def ocr_quality(result):
    text = result.get("text", "")
    return round(
        0.40 * result.get("confidence", 0)
        + 0.25 * min(1, sum(c.isalnum() for c in text) / 180)
        + 0.25 * min(1, len(label_fields(text)) / 6)
        + 0.10 * (header_type(text)["predicted"] != "unknown"),
        4,
    )


def edge_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        p = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        fraction = cv2.contourArea(p) / (image.shape[0] * image.shape[1])
        if len(p) != 4 or not cv2.isContourConvex(p) or not 0.30 <= fraction <= 0.99:
            continue
        p = p.reshape(4, 2).astype(np.float32)
        s = p.sum(axis=1)
        d = np.diff(p, axis=1).ravel()
        ordered = np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]])
        if len(np.unique(ordered, axis=0)) < 4:
            continue
        tl, tr, br, bl = ordered
        w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if min(w, h) < 160:
            continue
        target = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        return cv2.warpPerspective(
            image, cv2.getPerspectiveTransform(ordered, target), (w, h)
        ), ordered.tolist()
    return image, None


def rotate(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    matrix[0, 2] += nw / 2 - w / 2
    matrix[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(image, matrix, (nw, nh), borderValue=(255, 255, 255))


def prepare(image, mode="auto", ocr=None):
    warnings = []
    boundary = None
    if mode != "flat_scan":
        image, boundary = edge_crop(image)
        if boundary is None:
            warnings.append(
                "No reliable four-corner boundary; whole image retained. Check the crop."
            )
    trials = []
    rotation = 0
    if ocr:
        for turns in range(4):
            candidate = np.ascontiguousarray(np.rot90(image, turns))
            try:
                score = ocr_quality(ocr.read(resize_max(candidate, 1100)))
                trials.append({"rotation_degrees": turns * 90, "score": score})
            except Exception as error:
                trials.append(
                    {
                        "rotation_degrees": turns * 90,
                        "score": 0,
                        "error": type(error).__name__,
                    }
                )
        best = max(trials, key=lambda t: t["score"])
        if best["score"] > trials[0]["score"] + 0.08 and best["score"] > 0.30:
            rotation = best["rotation_degrees"]
            image = np.ascontiguousarray(np.rot90(image, rotation // 90))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(
        cv2.Canny(gray, 60, 160),
        1,
        np.pi / 180,
        threshold=70,
        minLineLength=max(60, image.shape[1] // 8),
        maxLineGap=20,
    )
    angles = []
    if lines is not None:
        for [[x1, y1, x2, y2]] in lines:
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) <= 12:
                angles.append(angle)
    deskew = 0
    if len(angles) >= 3 and np.std(angles) < 3:
        deskew = round(float(np.median(angles)), 2)
        if abs(deskew) >= 0.4:
            image = rotate(image, deskew)
        else:
            deskew = 0
    scale = min(2.5, max(1, 1600 / max(image.shape[:2])))
    if scale > 1:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    return image, {
        "boundary": boundary,
        "rotation_degrees": rotation,
        "deskew_degrees": deskew,
        "upscale_factor": round(scale, 3),
        "orientation_trials": trials,
        "warnings": warnings,
    }


def variants(image, count=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    shadow = cv2.divide(denoised, np.maximum(background, 1), scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(shadow)
    threshold = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 13
    )
    sharpened = cv2.addWeighted(
        clahe, 1.6, cv2.GaussianBlur(clahe, (0, 0), 1.2), -0.6, 0
    )
    return [("original", image)] + [
        (name, cv2.cvtColor(data, cv2.COLOR_GRAY2BGR))
        for name, data in [
            ("shadow_removed", shadow),
            ("clahe_denoised", clahe),
            ("adaptive_threshold", threshold),
            ("sharpened", sharpened),
        ]
    ][: count - 1]
