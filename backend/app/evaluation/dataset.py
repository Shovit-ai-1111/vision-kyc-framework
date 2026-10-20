import json
from pathlib import Path

SPLITS = {"train", "validation", "test"}


def load_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def validate_documents(rows, require_images=False):
    from app.nlp.fields import LABELS

    seen = set()
    subjects = {}
    images = {}
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate document id")
        seen.add(row["id"])
        split = row["split"]
        if split not in SPLITS:
            raise ValueError("Use train / validation / test splits")
        subject = row["subject_id"]
        if subject in subjects and subjects[subject] != split:
            raise ValueError(f"Identity leakage for {subject}")
        subjects[subject] = split
        if row.get("data_kind") not in ("synthetic_seed", "consented_real"):
            raise ValueError("Declare data_kind")
        if row["data_kind"] == "consented_real" and not row.get("consent_id"):
            raise ValueError("Consent reference required")
        if len(row["tokens"]) != len(row["labels"]) or not row["tokens"]:
            raise ValueError("One BIO label per OCR token required")
        if any(label not in LABELS for label in row["labels"]):
            raise ValueError("Invalid BIO label")
        for i, t in enumerate(row["tokens"]):
            if (
                len(t["box"]) != 4
                or t["box"][0] > t["box"][2]
                or t["box"][1] > t["box"][3]
            ):
                raise ValueError("Invalid bounding box")
            if any(
                v < 0 or v > (row["width"] if j % 2 == 0 else row["height"])
                for j, v in enumerate(t["box"])
            ):
                raise ValueError("Box outside image")
            label = row["labels"][i]
            if label.startswith("I-") and (
                i == 0
                or row["labels"][i - 1][2:] != label[2:]
                or row["labels"][i - 1] == "O"
            ):
                raise ValueError("Invalid BIO continuation")
        if require_images and not row.get("image"):
            raise ValueError("Document image path required")
        if row.get("image"):
            path = row["image"]
            if path in images and images[path] != split:
                raise ValueError("Image leakage across splits")
            images[path] = split
    if len({r["data_kind"] for r in rows}) != 1:
        raise ValueError("Keep synthetic and real evaluation datasets separate")
    return {
        "records": len(rows),
        "subjects": len(subjects),
        "splits": {s: sum(r["split"] == s for r in rows) for s in SPLITS},
    }
