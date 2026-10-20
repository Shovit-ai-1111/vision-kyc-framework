import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.core.config import Settings
from app.core.errors import KycError


class UnloadedRegistry:
    calibration = None

    def __init__(self, settings):
        self.settings = settings

    def require(self, *args):
        raise KycError("model_not_loaded", "Model not loaded: test registry", 503)

    def status(self):
        return {
            "ready": False,
            "components": {"face": {"loaded": False, "message": "Model not loaded"}},
            "ner_provenance": None,
        }

    def close(self):
        pass


@pytest.fixture
def client(tmp_path):
    s = Settings(
        data_dir=tmp_path,
        weights_dir=tmp_path / "weights",
        evaluations_dir=tmp_path / "eval",
        reviewer_key="r" * 32,
    )
    with TestClient(create_app(s, UnloadedRegistry(s))) as c:
        yield c


def create(client):
    return client.post(
        "/api/cases",
        json={
            "document_type": "pan",
            "claims": {"name": "Test Person", "document_number": "123456789"},
            "consent": True,
        },
    ).json()


def test_consent_and_access(client):
    assert (
        client.post(
            "/api/cases",
            json={
                "document_type": "pan",
                "claims": {"name": "Test Person", "document_number": "123456789"},
                "consent": False,
            },
        ).status_code
        == 422
    )
    case = create(client)
    path = "/api/cases/" + case["id"]
    headers = {"Authorization": "Bearer " + case["session_token"]}
    assert client.get(path).status_code == 401
    assert (
        client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    )
    assert client.get(path, headers=headers).status_code == 200
    assert client.get("/api/review/cases").status_code == 401
    assert (
        client.get(
            "/api/review/cases", headers={"X-Reviewer-Key": "r" * 32}
        ).status_code
        == 200
    )
    assert client.delete(path, headers=headers).status_code == 200
    assert client.get(path, headers=headers).status_code == 404


def test_missing_models_never_demo_success(client):
    case = create(client)
    headers = {"Authorization": "Bearer " + case["session_token"]}
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert (
        client.post(
            f"/api/cases/{case['id']}/challenge", json={}, headers=headers
        ).status_code
        == 503
    )
    image = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(image, format="PNG")
    response = client.post(
        f"/api/cases/{case['id']}/documents",
        headers=headers,
        data={"document_type": "pan", "side": "front"},
        files={"image": ("test.png", image.getvalue(), "image/png")},
    )
    assert (
        response.status_code == 503
        and response.json()["error"]["code"] == "model_not_loaded"
    )
    assert (
        client.post(
            f"/api/cases/{case['id']}/finish", json={}, headers=headers
        ).status_code
        == 409
    )


def test_reviewer_corrections_preserve_original_and_cas(client):
    case = create(client)
    headers = {"X-Reviewer-Key": "r" * 32}
    path = f"/api/review/cases/{case['id']}"
    payload = {
        "action": "CORRECT",
        "reviewer": "Reviewer One",
        "reason": "Corrected an OCR spelling error.",
        "corrections": {"name": "Corrected Person"},
        "expected_revision": 0,
    }
    result = client.post(path, headers=headers, json=payload)
    assert result.status_code == 200
    assert result.json()["claims"]["name"] == "Test Person"
    assert result.json()["corrected_fields"]["name"] == "Corrected Person"
    assert len(result.json()["review_history"]) == 1
    assert client.post(path, headers=headers, json=payload).status_code == 409
    payload.update(action="APPROVE", expected_revision=1)
    assert client.post(path, headers=headers, json=payload).status_code == 409


def test_reviewer_cannot_inject_fields(client):
    case = create(client)
    r = client.post(
        f"/api/review/cases/{case['id']}",
        headers={"X-Reviewer-Key": "r" * 32},
        json={
            "action": "CORRECT",
            "reviewer": "Reviewer",
            "reason": "invalid test correction",
            "corrections": {"result": "PASSED"},
            "expected_revision": 0,
        },
    )
    assert r.status_code == 422
