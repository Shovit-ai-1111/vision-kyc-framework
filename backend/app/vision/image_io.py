import base64
import binascii
import io
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import Settings
from app.core.errors import KycError


def resize_max(image: np.ndarray, size: int) -> np.ndarray:
    scale = min(1.0, size / max(image.shape[:2]))
    if scale == 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def decode_image(data: bytes, settings: Settings) -> np.ndarray:
    if not data or len(data) > settings.max_image_bytes:
        limit_mb = settings.max_image_bytes / (1024 * 1024)
        raise KycError(
            "image_size",
            f"Upload a non-empty image no larger than {limit_mb:g} MiB",
            413,
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise KycError(
                        "unsupported_image", "Use a JPEG, PNG, or WebP image", 415
                    )
                if image.width * image.height > settings.max_pixels:
                    raise KycError(
                        "image_dimensions", "Image exceeds the pixel limit", 413
                    )
                if min(image.size) < 160:
                    raise KycError(
                        "image_too_small",
                        "Image must be at least 160 pixels on each side",
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise KycError("animated_image", "Use a still image")
                image = ImageOps.exif_transpose(image).convert("RGB")
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except KycError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise KycError(
            "unreadable_image", "The image is damaged or cannot be decoded"
        ) from None


def decode_base64(value: str, settings: Settings) -> np.ndarray:
    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        if not separator or header not in {
            "data:image/jpeg;base64",
            "data:image/png;base64",
            "data:image/webp;base64",
        }:
            raise KycError(
                "invalid_image_encoding", "Use a JPEG, PNG, or WebP base64 data URL"
            )
    if len(value) > ((settings.max_image_bytes + 2) // 3) * 4:
        raise KycError("image_size", "Encoded image is too large", 413)
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise KycError("invalid_base64", "Image contains invalid base64 data") from None
    return decode_image(data, settings)


def preview_url(image: np.ndarray, max_size: int = 720) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", resize_max(image, max_size), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    if not ok:
        raise KycError("image_encode", "Unable to prepare the image preview", 500)
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
