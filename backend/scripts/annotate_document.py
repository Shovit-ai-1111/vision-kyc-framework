"""Create a draft token annotation from an image. A person must correct labels/text."""

import argparse, json
from pathlib import Path
from app.core.config import Settings
from app.vision.ocr import OCR
from app.vision.image_io import decode_image

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--id", required=True)
p.add_argument("--subject", required=True)
p.add_argument("--consent-id", required=True)
p.add_argument("--split", choices=["train", "validation", "test"], required=True)
p.add_argument("--type", choices=["citizenship", "passport", "pan", "national_id", "driving_license"], required=True)
p.add_argument("--side", choices=["front", "back"], default="front")
p.add_argument("--output", required=True)
args = p.parse_args()
s = Settings()
image = decode_image(Path(args.image).read_bytes(), s)
ocr = OCR(s).read(image)
import os

record = {
    "id": args.id,
    "subject_id": args.subject,
    "consent_id": args.consent_id,
    "data_kind": "consented_real",
    "split": args.split,
    "document_type": args.type,
    "side": args.side,
    "image": os.path.relpath(
        Path(args.image).resolve(), Path(args.output).resolve().parent
    ),
    "width": image.shape[1],
    "height": image.shape[0],
    "tokens": ocr["tokens"],
    "labels": ["O"] * len(ocr["tokens"]),
    "text": ocr["text"],
    "fields": {},
}
Path(args.output).write_text(json.dumps(record, ensure_ascii=False) + "\n")
print(
    "Draft saved. Correct transcription, BIO labels and fields before training/evaluation."
)
