import re
from app.nlp.fields import normalize

REQUIRED = {
    "citizenship": [
        "name",
        "father_name",
        "dob",
        "document_number",
        "district",
        "issue_date",
    ],
    "passport": ["name", "dob", "document_number", "issue_date"],
    "pan": ["name", "document_number"],
    "national_id": ["name", "dob", "document_number"],
    "driving_license": ["name", "dob", "document_number", "issue_date"],
}


def issue(code, message, severity="review", **extra):
    return {"code": code, "message": message, "severity": severity, **extra}


def classify(text):
    from app.nlp.nepali import header_type

    return header_type(text)


def mrz_checksum(value):
    table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    values = [0 if c == "<" else table.index(c) for c in value]
    return str(sum(v * [7, 3, 1][i % 3] for i, v in enumerate(values)) % 10)


def validate_mrz(text):
    lines = [re.sub(r"\s", "", line.upper()) for line in text.splitlines()]
    lines = [
        line for line in lines if len(line) == 44 and re.fullmatch("[A-Z0-9<]+", line)
    ]
    for i in range(len(lines) - 1):
        a, b = lines[i : i + 2]
        if not a.startswith("P<NPL"):
            continue
        pairs = {
            "document_number": (b[:9], b[9]),
            "dob": (b[13:19], b[19]),
            "expiry": (b[21:27], b[27]),
            "personal_number": (b[28:42], b[42]),
            "composite": (b[:10] + b[13:20] + b[21:43], b[43]),
        }
        checks = {
            k: mrz_checksum(v) == d
            or (k == "personal_number" and v == "<" * 14 and d == "<")
            for k, (v, d) in pairs.items()
        }
        surname, _, given = a[5:].partition("<<")
        return {
            "present": True,
            "valid": all(checks.values()),
            "checks": checks,
            "document_number": b[:9].replace("<", ""),
            "dob_yymmdd": b[13:19],
            "name": given.replace("<", " ").strip()
            + " "
            + surname.replace("<", " ").strip(),
        }
    return {"present": False, "valid": False, "checks": {}}


def compare_value(field, a, b):
    a, b = normalize(a), normalize(b)
    if field == "document_number":
        return (
            "match"
            if re.sub(r"[^a-z0-9]", "", a) == re.sub(r"[^a-z0-9]", "", b)
            else "mismatch"
        )
    if field in ("dob", "issue_date"):
        from app.nlp.nepali import convert_date

        da, db = convert_date(a), convert_date(b)
        if da["ad"] and db["ad"]:
            return "match" if da["ad"] == db["ad"] else "mismatch"
        return "uncertain"
    da = bool(re.search(r"[\u0900-\u097f]", a))
    db = bool(re.search(r"[\u0900-\u097f]", b))
    if da != db:
        return "uncertain"
    return "match" if a == b else "mismatch"


def check_case(case, settings):
    docs = case["documents"]
    issues = []
    comparisons = []
    primary = [d for d in docs if d["evidence"] == "primary"]
    fronts = [d for d in primary if d["side"] == "front"]
    if not fronts:
        issues.append(
            issue(
                "missing_front", "Upload the primary document portrait side.", "reject"
            )
        )
    if case["document_type"] == "citizenship" and not any(
        d["side"] == "back" for d in primary
    ):
        issues.append(
            issue("missing_back", "Citizenship requires front and back.", "review")
        )
    for d in docs:
        for item in d["issues"]:
            issues.append({**item, "document_id": d["id"]})
    for field in REQUIRED[case["document_type"]]:
        values = [
            (d, d["fields"].get(field, {}).get("selected"))
            for d in primary
            if d["fields"].get(field, {}).get("selected")
        ]
        if not values:
            issues.append(
                issue(
                    "missing_field",
                    f"Expected field not extracted: {field}.",
                    field=field,
                )
            )
            continue
        for d, value in values:
            if (
                value["confidence"] < settings.min_field_confidence
                or value["ocr_confidence"] < settings.min_ocr_confidence
            ):
                issues.append(
                    issue(
                        "low_field_confidence",
                        f"Low extraction/OCR confidence for {field}.",
                        field=field,
                        document_id=d["id"],
                    )
                )
        claimed = case["claims"].get(field, "")
        if claimed:
            for d, value in values:
                result = compare_value(field, value["value"], claimed)
                comparisons.append(
                    {
                        "field": field,
                        "source_a": d["id"],
                        "source_b": "entered",
                        "value_a": value["value"],
                        "value_b": claimed,
                        "result": result,
                    }
                )
                if result != "match":
                    issues.append(
                        issue(
                            "claim_" + result,
                            f"{field}: OCR and entered value {result}.",
                            "reject"
                            if result == "mismatch"
                            and value["confidence"] >= settings.min_field_confidence
                            and value["ocr_confidence"] >= settings.min_ocr_confidence
                            else "review",
                            field=field,
                        )
                    )
        else:
            issues.append(
                issue(
                    "missing_claim",
                    f"Enter {field} to compare it with OCR.",
                    field=field,
                )
            )
    # Compare all overlapping fields, including optional fields and secondary evidence.
    for i, a in enumerate(docs):
        for b in docs[i + 1 :]:
            for field in a["fields"]:
                if field == "document_number" and (
                    a["evidence"] != b["evidence"]
                    or a["requested_type"] != b["requested_type"]
                ):
                    continue
                av, bv = (
                    a["fields"][field]["selected"],
                    b["fields"].get(field, {}).get("selected"),
                )
                if not av or not bv:
                    continue
                result = compare_value(field, av["value"], bv["value"])
                comparisons.append(
                    {
                        "field": field,
                        "source_a": a["id"],
                        "source_b": b["id"],
                        "value_a": av["value"],
                        "value_b": bv["value"],
                        "result": result,
                    }
                )
                if result != "match":
                    confident = (
                        min(av["confidence"], bv["confidence"])
                        >= settings.min_field_confidence
                        and min(av["ocr_confidence"], bv["ocr_confidence"])
                        >= settings.min_ocr_confidence
                    )
                    issues.append(
                        issue(
                            "evidence_" + result,
                            f"{field}: evidence values {result}.",
                            "reject"
                            if result == "mismatch" and confident
                            else "review",
                            field=field,
                        )
                    )
    if case.get("applicant_corrections"):
        issues.append(
            issue(
                "applicant_corrections",
                "Applicant corrections require reviewer inspection; original extraction is retained.",
            )
        )
    return {"issues": issues, "comparisons": comparisons}
