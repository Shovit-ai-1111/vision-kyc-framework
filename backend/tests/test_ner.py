from pathlib import Path
from app.core.config import Settings
from app.nlp.fields import Extractor, features
from app.evaluation.dataset import load_jsonl, validate_documents


def test_shipped_crf_is_trained_and_predicts_held_out_tokens():
    root = Path(__file__).parents[1]
    rows = load_jsonl(root / "data/seed.jsonl")
    validate_documents(rows)
    weights = (
        root / "weights"
        if (root / "weights/ner-crf/model.crfsuite").is_file()
        else Path("/opt/seed")
    )
    model = Extractor(Settings(weights_dir=weights, ner_engine="crf"))
    record = next(
        r for r in rows if r["split"] == "test" and r["document_type"] == "citizenship"
    )
    fields = model.extract(record["tokens"], record["width"], record["height"])
    assert fields["name"]["selected"]["value"] == record["fields"]["name"]
    assert (
        fields["document_number"]["selected"]["value"]
        == record["fields"]["document_number"]
    )
    assert model.provenance["data_kind"] == "synthetic_seed"


def test_devanagari_feature():
    token = {"text": "नाम", "box": [0, 0, 10, 10], "line": 0}
    assert features([token], 0)["devanagari"] is True
