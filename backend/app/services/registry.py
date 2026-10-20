import json
import logging
from app.core.errors import KycError


class Registry:
    def __init__(self, settings):
        self.settings = settings
        self.items, self.states = {}, {}
        from app.vision.face_verifier import FaceEngine
        from app.vision.document_ocr import DocumentOCR
        from app.vision.liveness import Landmarker
        from app.nlp.fields import Extractor

        loaders = {
            "face": lambda: FaceEngine(settings),
            "ocr": lambda: DocumentOCR(settings),
            "ner": lambda: Extractor(settings),
            "landmarks": lambda: Landmarker(settings),
        }
        for name, loader in loaders.items():
            try:
                self.items[name] = loader()
                self.states[name] = {"loaded": True, "message": "Loaded"}
            except Exception as error:
                self.states[name] = {
                    "loaded": False,
                    "message": f"Model not loaded: {name}. {type(error).__name__}: {str(error)[:300]}",
                }
                logging.getLogger("kyc").error(
                    "model_not_loaded",
                    extra={
                        "event": "model_not_loaded",
                        "error_type": type(error).__name__,
                    },
                )
        self.calibration = None
        path = settings.weights_dir / "calibration.json"
        self.calibration_error = "No held-out face calibration installed."
        if path.is_file():
            try:
                result = json.loads(path.read_text())
                import hashlib

                digest = hashlib.sha256(
                    (
                        settings.weights_dir / f"facenet-{settings.face_weights}.pt"
                    ).read_bytes()
                ).hexdigest()
                if (
                    result["face_weights"] != settings.face_weights
                    or result["detector"] != settings.detector
                    or result["sha256"] != digest
                ):
                    raise ValueError(
                        "Calibration does not match selected model and detector."
                    )
                if (
                    result["split"] != "validation"
                    or result["subjects"] < 20
                    or result.get("study_subjects", 0) < 50
                    or result["genuine_count"] < 25
                    or result["impostor_count"] < 25
                    or not -1 <= result["threshold"] <= 1
                ):
                    raise ValueError(
                        "Calibration needs validation pairs, 20 validation identities and 50 total study identities."
                    )
                self.calibration = result
                self.calibration_error = None
            except Exception as error:
                self.calibration_error = f"Invalid calibration: {error}"

    def require(self, *names):
        for name in names:
            if name not in self.items:
                raise KycError("model_not_loaded", self.states[name]["message"], 503)
        return [self.items[name] for name in names]

    def status(self):
        return {
            "ready": all(v["loaded"] for v in self.states.values()),
            "components": self.states,
            "configuration": {
                "ocr": self.settings.ocr_engine,
                "ner": self.settings.ner_engine,
                "detector": self.settings.detector,
                "face_weights": self.settings.face_weights,
            },
            "ner_provenance": getattr(self.items.get("ner"), "provenance", None),
            "calibrated": self.calibration is not None,
            "calibration_message": self.calibration_error,
            "fallbacks": {"ocr": True, "face": False, "ner": False},
            "ocr_status": self.items["ocr"].status() if "ocr" in self.items else None,
            "assurance": "Academic evidence checks; issuer authenticity and certified PAD are not established.",
        }

    def close(self):
        if "landmarks" in self.items:
            self.items["landmarks"].close()
