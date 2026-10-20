from types import SimpleNamespace
import pytest
from app.core.config import Settings
from app.nlp.validation import classify, compare_value, validate_mrz, mrz_checksum
from app.services.decision import decide
from app.evaluation.metrics import edit_distance, ocr_counts, prf, binary_metrics, iou
from app.evaluation.dataset import validate_documents


@pytest.mark.parametrize(
    "text,kind",
    [
        ("नेपाल नागरिकता प्रमाणपत्र", "citizenship"),
        ("Nepal Passport", "passport"),
        ("स्थायी लेखा नम्बर", "pan"),
        ("a random face photo", "unknown"),
        ("Passport Citizenship", "unknown"),
    ],
)
def test_document_type(text, kind):
    assert classify(text)["predicted"] == kind


@pytest.mark.parametrize(
    "field,a,b,expected",
    [
        ("document_number", "१२३-४५६", "123456", "match"),
        ("name", "Ram Thapa", "Ram Sharma", "mismatch"),
        ("name", "राम थापा", "Ram Thapa", "uncertain"),
        ("dob", "२०५८-०२-११ BS", "2058-02-11 BS", "match"),
        ("dob", "2058-02-11 BS", "2058-02-11 AD", "uncertain"),
        ("dob", "2058-02-11", "2058-02-11", "uncertain"),
        ("district", "Kathmandu", " Kathmandu  ", "match"),
    ],
)
def test_comparison(field, a, b, expected):
    assert compare_value(field, a, b) == expected


def test_mrz_checksums():
    a = "P<NPLTEST<<PERSON".ljust(44, "<")
    number = "123456789"
    dob = "990101"
    expiry = "300101"
    optional = "<" * 14
    b = (
        number
        + mrz_checksum(number)
        + "NPL"
        + dob
        + mrz_checksum(dob)
        + "M"
        + expiry
        + mrz_checksum(expiry)
        + optional
        + "0"
    )
    b += mrz_checksum(b[:10] + b[13:20] + b[21:43])
    assert len(b) == 44
    assert validate_mrz(a + "\n" + b)["valid"]
    tampered = b[:5] + "0" + b[6:]
    assert not validate_mrz(a + "\n" + tampered)["valid"]


def registry(calibrated=True, real=True):
    return SimpleNamespace(
        settings=Settings(),
        calibration={"threshold": 0.7} if calibrated else None,
        status=lambda: {
            "ner_provenance": {
                "data_kind": "consented_real" if real else "synthetic_seed"
            }
        },
    )


@pytest.mark.parametrize(
    "passed,score,cal,real,status",
    [
        (True, 0.9, True, True, "PASSED_CHECKS"),
        (True, 0.3, True, True, "REJECTED_CHECKS"),
        (False, 0.9, True, True, "REJECTED_CHECKS"),
        (True, 0.9, False, True, "REVIEW_REQUIRED"),
        (True, 0.9, True, False, "REVIEW_REQUIRED"),
    ],
)
def test_decision_is_evidence_gated(passed, score, cal, real, status):
    result = decide(
        {"liveness": {"status": "PASSED" if passed else "FAILED"}},
        registry(cal, real),
        {"issues": []},
        score,
    )
    assert result["status"] == status
    assert result["authenticity"] == "NOT_VERIFIED_WITH_ISSUER"


def test_metrics_known_values():
    assert edit_distance("kitten", "sitting") == 3
    assert ocr_counts("abc", "axc")["character_errors"] == 1
    assert ocr_counts("१२३", "123")["character_errors"] == 3
    assert prf(2, 1, 1)["f1"] == pytest.approx(2 / 3)
    assert (
        binary_metrics([True, True, False, False], [True, False, True, False])["far"]
        == 0.5
    )
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1


def test_dataset_identity_leakage():
    r = {
        "id": "a",
        "subject_id": "p1",
        "split": "train",
        "data_kind": "synthetic_seed",
        "tokens": [{"text": "a", "box": [1, 1, 2, 2]}],
        "labels": ["O"],
        "width": 10,
        "height": 10,
    }
    with pytest.raises(ValueError, match="Identity leakage"):
        validate_documents([r, {**r, "id": "b", "split": "test"}])
