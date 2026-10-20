"""Cloud-first OCR with visible local fallback and independent cross-checks."""

import logging
from app.core.errors import KycError
from app.vision.ocr import OCR
from app.vision.gemini import GeminiReader, CloudUnavailable
from app.vision.preprocessing import variants, ocr_quality


class DocumentOCR:
    def __init__(self, settings):
        self.settings = settings
        self.local = None
        self.secondary = None
        self.load_messages = []
        for engine in dict.fromkeys([settings.ocr_engine, "tesseract"]):
            try:
                self.local = OCR(settings, engine)
                break
            except Exception as error:
                self.load_messages.append(
                    f"{engine} is unavailable ({type(error).__name__})."
                )
        if settings.ocr_secondary:
            try:
                self.secondary = OCR(
                    settings,
                    "easyocr"
                    if self.local and self.local.engine == "tesseract"
                    else "tesseract",
                )
            except Exception as error:
                self.load_messages.append(
                    f"Secondary local OCR unavailable ({type(error).__name__})."
                )
        self.cloud = GeminiReader(settings)
        if self.local is None and not (
            settings.gemini_enabled and settings.gemini_api_key.get_secret_value()
        ):
            raise RuntimeError(
                "Model not loaded: local OCR unavailable. Install Tesseract nep+eng or configure Gemini. "
                + " ".join(self.load_messages)
            )

    def status(self):
        return {
            "local_engine": self.local.engine if self.local else None,
            "secondary_engine": self.secondary.engine if self.secondary else None,
            "gemini_configured": bool(self.settings.gemini_api_key.get_secret_value()),
            "gemini_enabled": self.settings.gemini_enabled,
            "gemini_model": self.settings.gemini_model,
            "messages": self.load_messages,
            "requires_cloud_consent": True,
        }

    def read_document(self, image, cloud_consent=False):
        notices = list(self.load_messages)
        cloud = None
        if not self.settings.gemini_enabled:
            notices.append("Cloud OCR disabled; local OCR used.")
        elif not self.settings.gemini_api_key.get_secret_value():
            notices.append("GEMINI_API_KEY is missing; local OCR used.")
        elif not cloud_consent:
            notices.append("Cloud OCR was not permitted for this case; local OCR used.")
        else:
            try:
                cloud = self.cloud.read(image)
                notices.append(
                    "Gemini read the document. Local OCR cross-checks are shown below."
                )
            except CloudUnavailable as error:
                notices.append(str(error))
        options = variants(image, self.settings.ocr_variants)
        results = []
        trials = []
        work = (
            [(self.local, name, candidate) for name, candidate in options]
            if self.local
            else []
        )
        if self.secondary:
            work.append((self.secondary, "original", image))
        for reader, name, candidate in work:
            try:
                result = reader.read(candidate)
                result.update(variant=name, quality=ocr_quality(result))
                trials.append(
                    {
                        "engine": reader.engine,
                        "variant": name,
                        "score": result["quality"],
                        "confidence": result["confidence"],
                        "status": "ok",
                        "text": result["text"],
                    }
                )
                if result["tokens"]:
                    results.append(result)
            except Exception as error:
                logging.getLogger("kyc").warning(
                    "local_ocr_failed",
                    extra={
                        "event": "local_ocr_failed",
                        "error_type": type(error).__name__,
                    },
                )
                trials.append(
                    {
                        "engine": reader.engine,
                        "variant": name,
                        "status": "failed",
                        "error": type(error).__name__,
                    }
                )
        if not results and cloud is None:
            raise KycError(
                "ocr_unavailable",
                "No OCR engine produced readable text. "
                + " ".join(notices)
                + " Check local language packs, lighting and focus.",
                503 if self.local is None else 422,
            )
        best = (
            max(results, key=lambda r: r["quality"])
            if results
            else {
                "engine": "none",
                "tokens": [],
                "text": "",
                "confidence": 0,
                "variant": "original",
            }
        )
        if not results:
            notices.append(
                "Local OCR could not cross-check the cloud extraction; manual review required."
            )
        text = {
            **best,
            "engine": "gemini + " + best["engine"] if cloud else best["engine"],
            "local_engine": best["engine"],
            "text": cloud["raw_text"] if cloud else best["text"],
            "local_text": best["text"],
            "cloud": cloud,
            "notices": notices,
            "trials": trials,
            "cloud_used": cloud is not None,
            "confidence_kind": "uncalibrated engine score",
        }
        return (
            text,
            results,
            next(im for name, im in options if name == best["variant"]),
        )
