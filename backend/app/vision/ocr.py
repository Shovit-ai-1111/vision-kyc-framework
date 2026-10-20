from threading import RLock
import cv2
import numpy as np


class OCR:
    def __init__(self, settings, engine=None):
        self.engine = engine or settings.ocr_engine
        self.timeout = settings.ocr_timeout_seconds
        self.lock = RLock()
        if self.engine == "tesseract":
            import pytesseract

            self.reader = pytesseract
            languages = set(pytesseract.get_languages(config=""))
            if not {"eng", "nep"} <= languages:
                raise RuntimeError(
                    "Model not loaded: Tesseract eng + nep language packs are required."
                )
        elif self.engine == "easyocr":
            import easyocr

            self.reader = easyocr.Reader(
                ["ne", "en"],
                gpu=False,
                model_storage_directory=str(settings.weights_dir / "easyocr"),
                download_enabled=False,
                verbose=False,
            )
        else:
            raise ValueError("Unsupported OCR engine")

    def read(self, bgr):
        with self.lock:
            if self.engine == "tesseract":
                data = self.reader.image_to_data(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    lang="nep+eng",
                    config="--psm 11",
                    output_type=self.reader.Output.DICT,
                    timeout=self.timeout,
                )
                tokens = []
                for i, text in enumerate(data["text"]):
                    if not text.strip() or float(data["conf"][i]) < 0:
                        continue
                    x, y, w, h = (
                        int(data[k][i]) for k in ("left", "top", "width", "height")
                    )
                    tokens.append(
                        {
                            "text": text.strip(),
                            "box": [x, y, x + w, y + h],
                            "confidence": float(data["conf"][i]) / 100,
                        }
                    )
            else:
                tokens = []
                for points, text, confidence in self.reader.readtext(
                    bgr, detail=1, paragraph=False
                ):
                    points = np.asarray(points)
                    x1, y1 = points.min(axis=0)
                    x2, y2 = points.max(axis=0)
                    words = text.split()
                    # EasyOCR produces line boxes. Word sub-boxes are approximated and marked.
                    for i, word in enumerate(words):
                        tokens.append(
                            {
                                "text": word,
                                "box": [
                                    int(x1 + (x2 - x1) * i / len(words)),
                                    int(y1),
                                    int(x1 + (x2 - x1) * (i + 1) / len(words)),
                                    int(y2),
                                ],
                                "confidence": float(confidence),
                                "box_source": "estimated_from_line",
                            }
                        )
        lines = []
        for token in sorted(tokens, key=lambda t: (t["box"][1], t["box"][0])):
            cy = (token["box"][1] + token["box"][3]) / 2
            height = token["box"][3] - token["box"][1]
            match = next(
                (line for line in lines if abs(line["cy"] - cy) < max(8, height * 0.5)),
                None,
            )
            if match is None:
                match = {"cy": cy, "tokens": []}
                lines.append(match)
            match["tokens"].append(token)
        ordered = []
        for i, line in enumerate(sorted(lines, key=lambda l: l["cy"])):
            for token in sorted(line["tokens"], key=lambda t: t["box"][0]):
                token["line"] = i
                ordered.append(token)
        text = "\n".join(
            " ".join(t["text"] for t in ordered if t["line"] == i)
            for i in range(len(lines))
        )
        return {
            "engine": self.engine,
            "languages": ["ne", "en"],
            "text": text,
            "tokens": ordered,
            "confidence": float(np.mean([t["confidence"] for t in ordered]))
            if ordered
            else 0,
        }
