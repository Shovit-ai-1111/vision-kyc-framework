"""Merge evidence without converting engine confidence into identity probabilities."""

import copy
import re
from app.nlp.nepali import (
    ALL_FIELDS,
    digits,
    devanagari,
    label_fields,
    normalize_number,
    transliterate,
    convert_date,
    translated_text,
)

CANONICAL = (
    "document_type",
    "document_number",
    "full_name_nepali",
    "full_name_english",
    "dob_bs",
    "dob_ad",
    "gender",
    "district",
    "address",
    "father_name",
    "mother_name",
    "issue_date",
    "birth_place",
)


def same(a, b):
    return re.sub(r"\W", "", digits(a).casefold()) == re.sub(
        r"\W", "", digits(b).casefold()
    )


def merge_fields(text, local_results, ner, width, height, kind, min_confidence=0.70):
    pool = {f: [] for f in ALL_FIELDS}
    best_by_engine = {}
    for result in local_results:
        if (
            result["engine"] not in best_by_engine
            or result["quality"] > best_by_engine[result["engine"]]["quality"]
        ):
            best_by_engine[result["engine"]] = result
    for engine, result in best_by_engine.items():
        learned = ner.extract(result["tokens"], width, height)
        labelled = label_fields(result["text"], result["confidence"])
        for field in ALL_FIELDS:
            for c in learned.get(field, {}).get("candidates", []):
                pool[field].append(
                    {
                        **c,
                        "source": "ner",
                        "engine": engine,
                        "variant": result["variant"],
                        "score": min(c["confidence"], c["ocr_confidence"]) * 0.90,
                    }
                )
            for c in labelled.get(field, []):
                pool[field].append(
                    {
                        **c,
                        "engine": engine,
                        "variant": result["variant"],
                        "score": c["confidence"],
                    }
                )
    for candidate_result in local_results:
        best = best_by_engine[candidate_result["engine"]]
        if (
            candidate_result is best
            or candidate_result["quality"] < best["quality"] - 0.10
        ):
            continue
        for field, options in label_fields(
            candidate_result["text"], candidate_result["confidence"]
        ).items():
            if field not in pool:
                continue
            for c in options:
                pool[field].append(
                    {
                        **c,
                        "engine": candidate_result["engine"],
                        "variant": candidate_result["variant"],
                        "score": c["confidence"] * 0.95,
                        "source": "variant_label_mapping",
                    }
                )
    cloud = text.get("cloud")
    if cloud:
        for c in cloud["fields"]:
            if c["field"] not in pool or not c["original"].strip():
                continue
            supported = re.sub(r"\s+", "", digits(c["original"]).casefold()) in re.sub(
                r"\s+", "", digits(cloud["raw_text"]).casefold()
            )
            score = min(0.90 if supported else 0.45, c["confidence"])
            pool[c["field"]].append(
                {
                    "value": c["original"].strip(),
                    "english": c["english"].strip(),
                    "confidence": score,
                    "ocr_confidence": score,
                    "score": score,
                    "engine": "gemini",
                    "source": "gemini",
                    "calendar": c["calendar"],
                    "text_supported": supported,
                    "token_indices": [],
                    "box": None,
                }
            )
    result = {}
    for field, candidates in pool.items():
        for c in candidates:
            c["original"] = c["value"]
            if field == "document_number":
                numeric = normalize_number(c["value"], kind)
                c.update(
                    value=numeric["value"],
                    valid=numeric["valid"],
                    repaired=numeric["repaired"],
                )
                if not numeric["valid"]:
                    c["score"] = min(c["score"], 0.35)
                elif numeric["repaired"]:
                    c["score"] = min(c["score"], 0.65)
            if field in ("dob", "issue_date"):
                calendar = c.get("calendar")
                calendar = None if calendar == "unknown" else calendar
                c["date"] = convert_date(c["value"], calendar)
                c["value"] = digits(c["value"])
                if c["date"]["warning"]:
                    c["score"] = min(c["score"], 0.60)
        candidates.sort(
            key=lambda c: (c["score"], c["engine"] == "gemini"), reverse=True
        )
        selected = copy.deepcopy(candidates[0]) if candidates else None
        conflicts = []
        if selected:
            independent = [
                c
                for c in candidates
                if c["engine"] != selected["engine"] and c["score"] >= 0.45
            ]
            conflicts = [
                c for c in independent if not same(c["value"], selected["value"])
            ]
            conflicts += [
                c
                for c in candidates
                if c.get("variant") != selected.get("variant")
                and c["engine"] == selected["engine"]
                and c["score"] >= 0.60
                and not same(c["value"], selected["value"])
            ]
            agreements = {
                c["engine"] for c in independent if same(c["value"], selected["value"])
            }
            selected["agreement_engines"] = sorted(agreements)
            selected["confidence"] = min(
                0.94, selected["score"] + (0.04 if agreements else 0)
            )
            if conflicts:
                selected["confidence"] = min(0.65, selected["confidence"])
            if field == "document_number" and not agreements:
                selected["confidence"] = min(0.69, selected["confidence"])
                selected["review_reason"] = (
                    "Number requires confirmation without independent OCR agreement."
                )
            selected["needs_review"] = (
                selected["confidence"] < min_confidence
                or bool(conflicts)
                or selected.get("repaired", False)
            )
        result[field] = {
            "selected": selected,
            "candidates": candidates,
            "conflicts": [
                {"engine": c["engine"], "value": c["value"], "confidence": c["score"]}
                for c in conflicts
            ],
        }
    return result


def structure(fields, classification):
    values = {k: None for k in CANONICAL}
    confidence = {k: 0.0 for k in CANONICAL}
    provenance = {}
    originals = {}
    review = []
    values["document_type"] = classification["predicted"]
    confidence["document_type"] = classification.get("confidence") or 0

    def put(field, value, c, method="extracted"):
        values[field] = value or None
        confidence[field] = round(c.get("confidence", 0), 4) if value else 0
        provenance[field] = method
        if c.get("needs_review") or not value:
            review.append(field)

    for field, data in fields.items():
        c = data["selected"]
        if not c:
            continue
        raw = c.get("original", c["value"])
        originals[field] = raw
        english = c.get("english") or transliterate(c["value"])
        if field == "name":
            put("full_name_nepali", raw if devanagari(raw) else None, c)
            put(
                "full_name_english",
                english,
                c,
                "printed_or_cloud"
                if c.get("english") or not devanagari(raw)
                else "transliteration_suggestion",
            )
            if devanagari(raw):
                review.append("full_name_english")
        elif field in ("dob", "issue_date"):
            converted = c.get("date") or convert_date(raw)
            if field == "dob":
                put("dob_bs", converted["bs"], c, "calendar_conversion")
                put("dob_ad", converted["ad"], c, "calendar_conversion")
            else:
                put("issue_date", digits(raw), c)
                values["issue_date_bs"] = converted["bs"]
                values["issue_date_ad"] = converted["ad"]
                confidence["issue_date_bs"] = confidence["issue_date_ad"] = (
                    c["confidence"] if not converted["warning"] else 0
                )
            if converted["warning"]:
                review.append(field)
        else:
            put(
                field,
                c["value"] if field == "document_number" else english,
                c,
                "extracted"
                if not devanagari(raw)
                else "translation_or_transliteration",
            )
            if devanagari(raw) and field in (
                "father_name",
                "mother_name",
                "address",
                "birth_place",
                "district",
            ):
                review.append(field)
    return {
        **values,
        "original_nepali": originals,
        "field_confidence": confidence,
        "field_provenance": provenance,
        "review_fields": sorted(set(review)),
        "confidence_note": "Uncalibrated extraction scores. Latin name spelling must be confirmed.",
    }


def translate_document(text):
    return translated_text(text)
