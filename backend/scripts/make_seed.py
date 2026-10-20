"""Deterministic synthetic bilingual training text, NOT a real document dataset."""

import json, random
from pathlib import Path
from app.nlp.fields import LABEL_FIELDS

LABEL_TEXT = {
    "NAME": [
        ["Full Name :", "Name :", "Name of holder :"],
        ["नाम थर :", "पूरा नाम :", "नाम :"],
    ],
    "FATHER_NAME": [
        ["Father Name :", "Father's Name :"],
        ["बाबुको नाम :", "पिताको नाम :"],
    ],
    "DOB": [["Date of Birth :", "DOB :"], ["जन्म मिति :", "जन्ममिति :"]],
    "DOCUMENT_NUMBER": [
        ["Document Number :", "Citizenship No :", "PAN Number :", "Passport Number :"],
        ["प्रमाणपत्र नम्बर :", "नागरिकता नं :", "स्थायी लेखा नम्बर :", "राहदानी नम्बर :"],
    ],
    "DISTRICT": [["District :", "Issuing District :"], ["जिल्ला :", "जारी जिल्ला :"]],
    "ISSUE_DATE": [
        ["Issue Date :", "Date of Issue :"],
        ["जारी मिति :", "जारी गरेको मिति :"],
    ],
    "SURNAME": [["Surname :"], ["थर :"]],
    "GIVEN_NAMES": [["Given Names :"], ["पहिलो नाम :"]],
}


def build():
    rng = random.Random(42)
    rows = []
    en = [
        "Aarav",
        "Bibek",
        "Chirag",
        "Dinesh",
        "Eshan",
        "Farhan",
        "Gaurav",
        "Hari",
        "Ishan",
        "Jivan",
        "Kiran",
        "Lalit",
        "Milan",
        "Nabin",
        "Om",
        "Prabin",
        "Rajan",
        "Suman",
        "Tarun",
        "Umesh",
    ]
    ne = [
        "आरव",
        "विवेक",
        "चिराग",
        "दिनेश",
        "ईशान",
        "फरहान",
        "गौरव",
        "हरि",
        "इशान",
        "जीवन",
        "किरण",
        "ललित",
        "मिलन",
        "नवीन",
        "ओम",
        "प्रवीण",
        "राजन",
        "सुमन",
        "तरुण",
        "उमेश",
    ]
    for i in range(240):
        split = "train" if i < 160 else "validation" if i < 200 else "test"
        lang = i % 2
        # Name vocabulary disjoint by split, plus unique synthetic subject id.
        name_index = rng.choice(
            range(0, 12)
            if split == "train"
            else range(12, 16)
            if split == "validation"
            else range(16, 20)
        )
        first = (ne if lang else en)[name_index]
        surname = (
            rng.choice(["Sharma", "Thapa", "Rai", "Karki"])
            if not lang
            else rng.choice(["शर्मा", "थापा", "राई", "कार्की"])
        )
        kind = ["citizenship", "pan", "passport"][i % 3]
        values = {
            "NAME": first + " " + surname,
            "FATHER_NAME": ("Test Father " + str(i))
            if not lang
            else ("परीक्षण पिता " + str(i)),
            "DOB": f"{2055 + i % 5}-0{1 + i % 9}-15 BS",
            "DOCUMENT_NUMBER": str(100000000 + i)
            if kind == "pan"
            else (
                "PA" + str(100000 + i) if kind == "passport" else f"12-34-{10000 + i}"
            ),
            "DISTRICT": rng.choice(["Kathmandu", "Pokhara", "Butwal"])
            if not lang
            else rng.choice(["काठमाडौँ", "पोखरा", "बुटवल"]),
            "ISSUE_DATE": f"{2075 + i % 5}-0{1 + i % 9}-20 BS",
            "SURNAME": surname,
            "GIVEN_NAMES": first,
        }
        entries = (
            ["DOCUMENT_NUMBER", "NAME"]
            if kind == "pan"
            else ["NAME", "DOB", "DOCUMENT_NUMBER", "ISSUE_DATE"]
            if kind == "passport"
            else [
                "NAME",
                "FATHER_NAME",
                "DOB",
                "DOCUMENT_NUMBER",
                "DISTRICT",
                "ISSUE_DATE",
            ]
        )
        if kind == "passport" and i % 4 == 0:
            entries.remove("NAME")
            entries += ["SURNAME", "GIVEN_NAMES"]
        rng.shuffle(entries)
        header = {
            "citizenship": "नेपाल नागरिकता" if lang else "Nepal Citizenship",
            "passport": "नेपाल राहदानी" if lang else "Nepal Passport",
            "pan": "स्थायी लेखा" if lang else "Permanent Account",
        }[kind]
        lines = [(header, None), ("SAMPLE NOT VALID", None)]
        for entity in entries:
            lines.append((rng.choice(LABEL_TEXT[entity][lang]), entity))
        lines += [("Signature : Registrar", None)]
        tokens = []
        labels = []
        text = []
        gold = {}
        for line, (prefix, entity) in enumerate(lines):
            words = prefix.split()
            value = values[entity].split() if entity else []
            text.append(" ".join(words + value))
            x = 45
            for wi, w in enumerate(words + value):
                width = max(13, len(w) * 11)
                tokens.append(
                    {
                        "text": w,
                        "box": [x, 40 + line * 48, x + width, 70 + line * 48],
                        "line": line,
                        "confidence": 1.0,
                    }
                )
                x += width + 10
                labels.append(
                    "O"
                    if wi < len(words)
                    else ("B-" if wi == len(words) else "I-") + entity
                )
            if entity:
                gold[LABEL_FIELDS[entity]] = values[entity]
        if "surname" in gold:
            gold["name"] = gold.pop("given_names") + " " + gold.pop("surname")
        rows.append(
            {
                "id": f"seed-{i:03d}",
                "subject_id": f"synthetic-person-{i:03d}",
                "split": split,
                "data_kind": "synthetic_seed",
                "document_type": kind,
                "side": "front",
                "width": 1000,
                "height": 650,
                "tokens": tokens,
                "labels": labels,
                "fields": gold,
                "text": "\n".join(text),
                "image": None,
            }
        )
    return rows


if __name__ == "__main__":
    path = Path("data/seed.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in build()) + "\n"
    )
    print(f"Wrote {path}")
