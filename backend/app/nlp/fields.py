"""Learned BIO sequence labelling, with OCR confidence and layout features."""

import json
import re
import unicodedata
from threading import RLock
import numpy as np
from app.models.schemas import FIELDS

LABEL_FIELDS = {
    "NAME": "name",
    "FATHER_NAME": "father_name",
    "DOB": "dob",
    "DOCUMENT_NUMBER": "document_number",
    "DISTRICT": "district",
    "ISSUE_DATE": "issue_date",
    "SURNAME": "surname",
    "GIVEN_NAMES": "given_names",
}
LABELS = ["O"] + [f"{prefix}-{name}" for name in LABEL_FIELDS for prefix in ("B", "I")]
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize(value):
    return " ".join(
        unicodedata.normalize("NFC", value).translate(DIGITS).casefold().split()
    ).strip(" :।")


def features(tokens, i, width=1000, height=1000):
    token = tokens[i]
    word = re.sub(r"\d", "#", normalize(token["text"]))
    box = token["box"]
    f = {
        "bias": 1.0,
        "word": word,
        "suffix2": word[-2:],
        "suffix3": word[-3:],
        "numeric": any(c.isdigit() for c in word),
        "devanagari": any("\u0900" <= c <= "\u097f" for c in word),
        "x": round(box[0] / width, 1),
        "y": round(box[1] / height, 1),
        "colon": ":" in word,
        "BOS": i == 0,
        "EOS": i == len(tokens) - 1,
    }
    line = [
        re.sub(r"\d", "#", normalize(t["text"]))
        for t in tokens
        if t["line"] == token["line"]
    ]
    for n, w in enumerate(line[:4]):
        f[f"line_prefix_{n}"] = w
    for w in line[:12]:
        f["line_contains=" + w] = True
    for offset in (-3, -2, -1, 1, 2, 3):
        j = i + offset
        if 0 <= j < len(tokens):
            f[f"w{offset}"] = re.sub(r"\d", "#", normalize(tokens[j]["text"]))
            f[f"same_line{offset}"] = tokens[j]["line"] == token["line"]
    return f


def decode_spans(tokens, labels, confidences):
    spans = []
    current = None
    for i, (token, label, confidence) in enumerate(zip(tokens, labels, confidences)):
        if label == "O":
            current = None
            continue
        prefix, entity = label.split("-", 1)
        if entity not in LABEL_FIELDS:
            raise ValueError(f"Unexpected model label {entity}")
        if (
            prefix == "B"
            or current is None
            or current["entity"] != entity
            or token["line"] != tokens[current["indices"][-1]]["line"]
        ):
            current = {"entity": entity, "indices": [], "scores": []}
            spans.append(current)
        current["indices"].append(i)
        current["scores"].append(float(confidence))
    result = {}
    for span in spans:
        selected = [tokens[i] for i in span["indices"]]
        value = " ".join(t["text"] for t in selected).strip(" :")
        if not value:
            continue
        candidate = {
            "value": value,
            "confidence": float(np.mean(span["scores"])),
            "ocr_confidence": float(np.mean([t["confidence"] for t in selected])),
            "token_indices": span["indices"],
            "box": [
                min(t["box"][0] for t in selected),
                min(t["box"][1] for t in selected),
                max(t["box"][2] for t in selected),
                max(t["box"][3] for t in selected),
            ],
        }
        result.setdefault(LABEL_FIELDS[span["entity"]], []).append(candidate)
    if result.get("surname") and result.get("given_names"):
        s, g = result["surname"][0], result["given_names"][0]
        result.setdefault("name", []).append(
            {
                **g,
                "value": g["value"] + " " + s["value"],
                "confidence": min(s["confidence"], g["confidence"]),
                "ocr_confidence": min(s["ocr_confidence"], g["ocr_confidence"]),
                "token_indices": g["token_indices"] + s["token_indices"],
            }
        )
    return {
        field: {
            "candidates": result.get(field, []),
            "selected": max(result[field], key=lambda x: x["confidence"])
            if result.get(field)
            else None,
        }
        for field in FIELDS
    }


class Extractor:
    def __init__(self, settings):
        self.settings = settings
        self.lock = RLock()
        directory = settings.weights_dir / (
            "ner-crf" if settings.ner_engine == "crf" else "ner-transformer"
        )
        meta = directory / "provenance.json"
        if not meta.is_file():
            raise FileNotFoundError(
                f"Model not loaded: {meta}. Train the selected NER engine."
            )
        self.provenance = json.loads(meta.read_text())
        if self.provenance.get("labels") != LABELS or not self.provenance.get(
            "trained"
        ):
            raise ValueError("NER checkpoint lacks the trained KYC label contract.")
        if settings.ner_engine == "crf":
            import pycrfsuite

            self.model = pycrfsuite.Tagger()
            self.model.open(str(directory / "model.crfsuite"))
        else:
            from transformers import AutoTokenizer, AutoModelForTokenClassification

            self.tokenizer = AutoTokenizer.from_pretrained(
                directory, local_files_only=True, use_fast=True
            )
            self.model = (
                AutoModelForTokenClassification.from_pretrained(
                    directory, local_files_only=True
                )
                .eval()
                .to(settings.device)
            )
            if [self.model.config.id2label[i] for i in range(len(LABELS))] != LABELS:
                raise ValueError(
                    "Unexpected NER labels; refusing an unfine-tuned/general NER checkpoint."
                )

    def extract(self, tokens, width, height):
        if not tokens:
            return decode_spans([], [], [])
        with self.lock:
            if self.settings.ner_engine == "crf":
                self.model.set(
                    [features(tokens, i, width, height) for i in range(len(tokens))]
                )
                labels = self.model.tag()
                confidence = [
                    self.model.marginal(label, i) for i, label in enumerate(labels)
                ]
            else:
                import torch

                enc = self.tokenizer(
                    [t["text"] for t in tokens],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    stride=64,
                    return_overflowing_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                mapping = [enc.word_ids(i) for i in range(len(enc["input_ids"]))]
                inputs = {
                    k: v.to(self.settings.device)
                    for k, v in enc.items()
                    if k != "overflow_to_sample_mapping"
                }
                if self.model.config.model_type == "lilt":
                    boxes = []
                    for ids in mapping:
                        boxes.append(
                            [
                                [
                                    max(
                                        0,
                                        min(
                                            1000,
                                            int(
                                                v
                                                / (width if j % 2 == 0 else height)
                                                * 1000
                                            ),
                                        ),
                                    )
                                    for j, v in enumerate(tokens[w]["box"])
                                ]
                                if w is not None
                                else [0, 0, 0, 0]
                                for w in ids
                            ]
                        )
                    inputs["bbox"] = torch.tensor(boxes, device=self.settings.device)
                with torch.inference_mode():
                    probs = self.model(**inputs).logits.softmax(-1).cpu().numpy()
                labels = ["O"] * len(tokens)
                confidence = [0.0] * len(tokens)
                for window, ids in enumerate(mapping):
                    seen = set()
                    for index, word_id in enumerate(ids):
                        if word_id is None or word_id in seen:
                            continue
                        seen.add(word_id)
                        label = int(probs[window, index].argmax())
                        score = float(probs[window, index, label])
                        if score > confidence[word_id]:
                            labels[word_id] = LABELS[label]
                            confidence[word_id] = score
        return decode_spans(tokens, labels, confidence)
