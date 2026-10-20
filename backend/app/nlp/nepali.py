"""Traceable bilingual labels, numeric normalization and exact calendar conversion."""

from datetime import date
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata

MAPPING = json.loads(Path(__file__).with_name("nepali_labels.json").read_text())
ALL_FIELDS = (
    "name",
    "father_name",
    "dob",
    "document_number",
    "district",
    "issue_date",
    "mother_name",
    "gender",
    "address",
    "birth_place",
)
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def digits(value):
    return unicodedata.normalize("NFC", str(value)).translate(DIGITS)


def key(value):
    return re.sub(r"[\s.:：।\-()/]+", "", digits(value).casefold()).replace("ं", "ँ")


def devanagari(value):
    return bool(re.search(r"[\u0900-\u097f]", value))


def header_type(text):
    matches = {kind: [] for kind in MAPPING["headers"]}
    for line in text.splitlines():
        words = line.split()
        for kind, aliases in MAPPING["headers"].items():
            for alias in aliases:
                target = key(alias)
                score = 0
                if target in key(line):
                    score = 1
                elif len(target) >= 7:
                    n = len(alias.split())
                    for size in {max(1, n - 1), n, n + 1}:
                        for i in range(max(0, len(words) - size + 1)):
                            score = max(
                                score,
                                SequenceMatcher(
                                    None, target, key(" ".join(words[i : i + size]))
                                ).ratio(),
                            )
                if score >= 0.86:
                    matches[kind].append({"header": alias, "score": round(score, 3)})
    positive = [k for k, v in matches.items() if v]
    predicted = positive[0] if len(positive) == 1 else "unknown"
    return {
        "predicted": predicted,
        "method": "bilingual fuzzy heading rules",
        "evidence": matches,
        "confidence": max((v["score"] for v in matches.get(predicted, [])), default=0),
    }


def label_match(line):
    line = line.strip()
    matches = []
    for field, aliases in MAPPING["labels"].items():
        for alias in aliases:
            pattern = (
                r"^"
                + r"\s*".join(re.escape(p) for p in alias.split())
                + r"(?=\s|[:：।.]|$)"
            )
            m = re.match(pattern, line, re.I)
            if m:
                matches.append((m.end(), field, 1.0))
    if matches:
        end, field, score = max(matches)
        return field, line[end:].lstrip(" .:：।\t"), score
    parts = re.split(r"[:：]", line, maxsplit=1)
    candidates = [(parts[0], parts[1])] if len(parts) == 2 else []
    words = line.split()
    candidates += [
        (" ".join(words[:n]), " ".join(words[n:])) for n in range(1, min(6, len(words)))
    ]
    best = None
    for label, value in candidates:
        for field, aliases in MAPPING["labels"].items():
            for alias in aliases:
                if len(key(alias)) < 5:
                    continue
                score = SequenceMatcher(None, key(label), key(alias)).ratio()
                if score >= 0.86 and (best is None or score > best[2]):
                    best = field, value.strip(), score
    return best


def label_fields(text, confidence=0.7):
    lines, result = text.splitlines(), {}
    for i, line in enumerate(lines):
        match = label_match(line)
        if not match:
            continue
        field, value, similarity = match
        if (
            not value
            and i + 1 < len(lines)
            and not label_match(lines[i + 1])
            and header_type(lines[i + 1])["predicted"] == "unknown"
        ):
            value = lines[i + 1].strip()
        if not value or value.strip(" ._-।:") == "" or len(value) > 240:
            continue
        calendar = (
            "BS"
            if field in ("dob", "issue_date") and re.search(r"[वब]ि\s*\.?\s*सं", line)
            else None
        )
        result.setdefault(field, []).append(
            {
                "value": value,
                "calendar": calendar,
                "confidence": min(0.94, confidence * similarity),
                "ocr_confidence": confidence,
                "source": "label_mapping",
                "label_similarity": similarity,
                "token_indices": [],
                "box": None,
            }
        )
    if result.get("surname") and result.get("given_names"):
        result["name"] = [
            {
                **result["given_names"][0],
                "value": result["given_names"][0]["value"]
                + " "
                + result["surname"][0]["value"],
            }
        ]
    return result


def normalize_number(value, kind):
    original = value
    value = digits(value).upper().strip(" .:।")
    value = re.sub(r"[–—−]", "-", value)
    value = re.sub(r"\s*([/\-])\s*", r"\1", value)
    prefix = ""
    if kind == "passport":
        m = re.match(r"^[A-Z]{1,2}(?=[0-9])", value)
        if m:
            prefix, value = m.group(), value[m.end() :]
    repaired = value.translate(str.maketrans({"O": "0", "I": "1", "L": "1", "|": "1"}))
    value = prefix + re.sub(r"\s+", "", repaired)
    valid = any(
        re.fullmatch(p, value) for p in MAPPING["number_patterns"].get(kind, [])
    )
    changed = bool(re.search("[OIL|]", digits(original).upper()[len(prefix) :]))
    return {"value": value, "valid": valid, "repaired": changed, "original": original}


CONSONANTS = dict(
    zip(
        "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
        [
            "k",
            "kh",
            "g",
            "gh",
            "ng",
            "ch",
            "chh",
            "j",
            "jh",
            "ny",
            "t",
            "th",
            "d",
            "dh",
            "n",
            "t",
            "th",
            "d",
            "dh",
            "n",
            "p",
            "ph",
            "b",
            "bh",
            "m",
            "y",
            "r",
            "l",
            "w",
            "sh",
            "sh",
            "s",
            "h",
        ],
    )
)
VOWELS = dict(
    zip("अआइईउऊऋएऐओऔ", ["a", "a", "i", "i", "u", "u", "ri", "e", "ai", "o", "au"])
)
MARKS = dict(zip("ािीुूृेैोौ", ["a", "i", "i", "u", "u", "ri", "e", "ai", "o", "au"]))


def transliterate(value):
    value = digits(value)
    if value in MAPPING["translations"]:
        return MAPPING["translations"][value]
    if not devanagari(value):
        return value

    def word(w):
        if w in MAPPING["translations"]:
            return MAPPING["translations"][w]
        output = []
        for i, c in enumerate(w):
            following = w[i + 1] if i + 1 < len(w) else ""
            if c in CONSONANTS:
                output.append(CONSONANTS[c])
                if following and following not in MARKS and following != "्":
                    output.append("a")
            elif c in VOWELS:
                output.append(VOWELS[c])
            elif c in MARKS:
                output.append(MARKS[c])
            elif c in "ंँ":
                output.append("n")
            elif c == "ः":
                output.append("h")
            elif c not in "़्":
                output.append(c)
        return "".join(output).capitalize()

    return " ".join(word(w) for w in value.split())


def convert_date(value, calendar=None):
    value = digits(value)
    result = {"bs": None, "ad": None, "calendar": calendar, "warning": None}
    explicit_bs = bool(re.search(r"\bbs\b|[वब]ि\s*\.?\s*सं", value, re.I))
    explicit_ad = bool(re.search(r"\bad\b|ई\s*\.?\s*सं", value, re.I))
    if explicit_bs and explicit_ad:
        result["warning"] = "Conflicting BS/AD calendar labels; confirm the date."
        return result
    calendar = calendar or ("BS" if explicit_bs else "AD" if explicit_ad else None)
    if calendar not in ("BS", "AD"):
        result["warning"] = (
            "Calendar is ambiguous; identify BS or AD before conversion."
        )
        return result
    if (calendar == "BS" and explicit_ad) or (calendar == "AD" and explicit_bs):
        result["warning"] = "Conflicting calendar information."
        return result
    parts = re.findall(r"\d+", value)
    if len(parts) != 3 or len(parts[0]) != 4:
        result["warning"] = "Use an unambiguous YYYY-MM-DD date with its calendar."
        return result
    try:
        import nepali_datetime

        y, m, d = map(int, parts)
        if calendar == "BS":
            bs = nepali_datetime.date(y, m, d)
            ad = bs.to_datetime_date()
        else:
            ad = date(y, m, d)
            bs = nepali_datetime.date.from_datetime_date(ad)
        result.update(bs=bs.isoformat(), ad=ad.isoformat(), calendar=calendar)
    except (ValueError, OverflowError, TypeError):
        result["warning"] = (
            "Invalid date or date outside supported BS conversion tables."
        )
    return result


def translated_text(text):
    output = []
    for line in text.splitlines():
        match = label_match(line)
        if match:
            field, value, _ = match
            output.append(field.replace("_", " ").title() + ": " + transliterate(value))
        else:
            output.append(transliterate(line))
    return "\n".join(output)
