"""Server framing guidance in mirrored-preview coordinates."""


def face_guidance(box, brightness, sharpness, step="NEUTRAL", nose_offset=0):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    cx, cy = 1 - (x1 + x2) / 2, (y1 + y2) / 2
    lighting = 45 <= brightness <= 220
    size = 0.22 <= width <= 0.58 and 0.30 <= height <= 0.82
    centered = (
        abs(cx - 0.5) <= (0.18 if step.startswith("TURN_") else 0.12)
        and abs(cy - 0.48) <= 0.16
    )
    focus = sharpness >= 18
    if width < 0.22 or height < 0.30:
        prompt = "Move closer"
    elif width > 0.58 or height > 0.82:
        prompt = "Move back"
    elif cx < 0.38:
        prompt = "Move right"
    elif cx > 0.62:
        prompt = "Move left"
    elif cy < 0.32:
        prompt = "Lower your face slightly"
    elif cy > 0.64:
        prompt = "Raise your face slightly"
    elif not lighting:
        prompt = "Face a light source" if brightness < 45 else "Move away from glare"
    elif not focus:
        prompt = "Hold still"
    elif step in ("NEUTRAL", "CENTER") and abs(nose_offset) > 0.18:
        prompt = "Look straight"
    else:
        prompt = "Good, hold still…"
    ready = size and centered and lighting and focus
    return {
        "prompt": prompt,
        "color": "green" if ready else "yellow",
        "ready": ready,
        "face_detected": True,
        "face_count": 1,
        "lighting_ok": lighting,
        "size_ok": size,
        "centered": centered,
        "focus_ok": focus,
        "box": box,
        "preview_mirrored": True,
    }


def no_face_guidance(message="No face detected", count=0):
    return {
        "prompt": "Only one person in view"
        if count > 1
        else "Position your face inside the oval",
        "color": "red",
        "ready": False,
        "face_detected": count > 0,
        "face_count": count,
        "lighting_ok": None,
        "size_ok": False,
        "centered": False,
        "focus_ok": None,
        "message": message,
        "preview_mirrored": True,
    }
