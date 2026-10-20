import json
import secrets
from typing import Literal
from fastapi import APIRouter, Depends, Header, Request, UploadFile, File, Form
from app.core.errors import KycError
from app.models.schemas import (
    CaseInput,
    FrameInput,
    ReviewInput,
    FIELDS,
    DocumentType,
    ApplicantCorrectionInput,
)
from app.services.pipeline import public_case
from app.nlp.validation import check_case

router = APIRouter(prefix="/api")


def pipeline(request: Request):
    return request.app.state.pipeline


def token(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise KycError("unauthorized", "A session token is required.", 401)
    return authorization[7:]


def reviewer(request: Request, x_reviewer_key: str = Header(default="")):
    expected = request.app.state.settings.reviewer_key
    if len(expected) < 24:
        raise KycError(
            "reviewer_not_configured",
            "Set KYC_REVIEWER_KEY using the setup script.",
            503,
        )
    if not secrets.compare_digest(expected, x_reviewer_key):
        raise KycError("reviewer_unauthorized", "Invalid reviewer key.", 401)


@router.get("/models")
def models(request: Request):
    return request.app.state.registry.status()


@router.post("/cases", status_code=201)
def create(payload: CaseInput, p=Depends(pipeline)):
    case, key = p.store.create(payload.model_dump())
    return {**public_case(case), "session_token": key}


@router.get("/cases/{case_id}")
def get_case(case_id: str, key=Depends(token), p=Depends(pipeline)):
    return public_case(p.store.get(case_id, key))


@router.delete("/cases/{case_id}")
def delete_case(case_id: str, key=Depends(token), p=Depends(pipeline)):
    with p.lock:
        p.store.get(case_id, key)
        p.store.delete(case_id)
        p.challenges.pop(case_id, None)
    return {"deleted": True}


@router.post("/cases/{case_id}/documents")
def upload(
    case_id: str,
    image: UploadFile = File(...),
    document_type: DocumentType = Form(...),
    side: Literal["front", "back"] = Form(...),
    evidence: Literal["primary", "secondary"] = Form("primary"),
    input_mode: Literal["auto", "flat_scan", "photograph"] = Form("auto"),
    key=Depends(token),
    p=Depends(pipeline),
):
    data = image.file.read(p.settings.max_image_bytes + 1)
    return p.add_document(case_id, key, data, document_type, side, evidence, input_mode)


@router.delete("/cases/{case_id}/documents/{document_id}")
def remove_document(
    case_id: str, document_id: str, key=Depends(token), p=Depends(pipeline)
):
    with p.lock:
        case = p.store.get(case_id, key)
        if case["result"]:
            raise KycError("case_complete", "Start a new case to change evidence.", 409)
        case["documents"] = [d for d in case["documents"] if d["id"] != document_id]
        p.challenges.pop(case_id, None)
        case["liveness"] = None
        case["consistency"] = check_case(case, p.settings)
        p.store.save(case)
        return public_case(case)


@router.post("/cases/{case_id}/challenge")
def start(case_id: str, key=Depends(token), p=Depends(pipeline)):
    return p.start(case_id, key)


@router.post("/cases/{case_id}/challenge/{challenge_id}/frames")
def frame(
    case_id: str,
    challenge_id: str,
    payload: FrameInput,
    key=Depends(token),
    p=Depends(pipeline),
):
    return p.frame(case_id, key, challenge_id, payload)


@router.post("/cases/{case_id}/finish")
def finish(case_id: str, key=Depends(token), p=Depends(pipeline)):
    return p.finish(case_id, key)


@router.get("/review/cases", dependencies=[Depends(reviewer)])
def review_list(p=Depends(pipeline)):
    return [
        {
            "id": c["id"],
            "document_type": c["document_type"],
            "created_at": c["created_at"],
            "revision": c["revision"],
            "result": c.get("result"),
            "review": c.get("review"),
        }
        for c in p.store.list()
    ]


@router.get("/review/cases/{case_id}", dependencies=[Depends(reviewer)])
def review_get(case_id: str, p=Depends(pipeline)):
    return public_case(p.store.get(case_id, reviewer=True))


@router.post("/review/cases/{case_id}", dependencies=[Depends(reviewer)])
def review(case_id: str, payload: ReviewInput, p=Depends(pipeline)):
    with p.lock:
        case = p.store.get(case_id, reviewer=True)
        if case["revision"] != payload.expected_revision:
            raise KycError(
                "revision_conflict", "Another update occurred. Reload the case.", 409
            )
        if any(
            k not in FIELDS or not isinstance(v, str) or len(v) > 200
            for k, v in payload.corrections.items()
        ):
            raise KycError(
                "invalid_correction",
                "Only the six KYC fields can be corrected, maximum 200 characters each.",
            )
        if payload.action == "CORRECT" and not payload.corrections:
            raise KycError("empty_correction", "Enter at least one correction.")
        if payload.action == "APPROVE" and (
            not case["result"]
            or not case.get("liveness")
            or case["liveness"]["status"] != "PASSED"
        ):
            raise KycError(
                "incomplete_evidence",
                "Complete the live session and automated evidence checks before reviewer approval.",
                409,
            )
        import time

        event = {**payload.model_dump(), "at": time.time()}
        case["review_history"].append(event)
        case.setdefault("corrected_fields", {}).update(payload.corrections)
        if payload.action in ("APPROVE", "REJECT"):
            case["review"] = event
        else:
            case["review"] = (
                None  # Any correction invalidates an earlier reviewer disposition.
            )
        p.store.save(case)
        return public_case(case)


@router.delete("/review/cases/{case_id}", dependencies=[Depends(reviewer)])
def review_delete(case_id: str, p=Depends(pipeline)):
    with p.lock:
        p.store.delete(case_id)
        p.challenges.pop(case_id, None)
    return {"deleted": True}


@router.get("/evaluations")
def evaluations(request: Request):
    root = request.app.state.settings.evaluations_dir
    summaries = []
    for path in sorted(root.glob("*/summary.json")):
        data = json.loads(path.read_text())
        # Public endpoint exposes only aggregate metrics, never manifests or source images.
        summaries.append(
            {
                "run": path.parent.name,
                **{
                    k: v
                    for k, v in data.items()
                    if k
                    in ("kind", "data_kind", "metrics", "charts", "counts", "notes")
                },
            }
        )
    return {
        "runs": summaries,
        "note": "Empty panels mean not measured. Synthetic results are explicitly labelled.",
    }


@router.post("/cases/{case_id}/documents/{document_id}/corrections")
def applicant_correct(
    case_id: str,
    document_id: str,
    payload: ApplicantCorrectionInput,
    key=Depends(token),
    p=Depends(pipeline),
):
    import time
    from app.nlp.structured import CANONICAL
    from app.nlp.nepali import normalize_number, convert_date

    with p.lock:
        case = p.store.get(case_id, key)
        if case["result"]:
            raise KycError("case_complete", "Completed cases are immutable.", 409)
        if case["revision"] != payload.expected_revision:
            raise KycError(
                "revision_conflict", "Another update occurred. Reload the case.", 409
            )
        doc = next((d for d in case["documents"] if d["id"] == document_id), None)
        if doc is None:
            raise KycError("document_not_found", "Document not found.", 404)
        values = {k: v.strip() for k, v in payload.corrections.items()}
        if any(
            k not in CANONICAL or k == "document_type" or len(v) > 240
            for k, v in values.items()
        ):
            raise KycError(
                "invalid_correction",
                "Only displayed fields can be corrected, maximum 240 characters.",
                422,
            )
        if values.get("document_number"):
            n = normalize_number(values["document_number"], doc["requested_type"])
            if not n["valid"] or n["repaired"]:
                raise KycError(
                    "invalid_number",
                    "Enter a valid document number using unambiguous digits.",
                    422,
                )
            values["document_number"] = n["value"]
        converted = []
        for name, calendar in (("dob_bs", "BS"), ("dob_ad", "AD")):
            if values.get(name):
                d = convert_date(values[name], calendar)
                if d["warning"]:
                    raise KycError("invalid_date", d["warning"], 422)
                converted.append(d)
        if len(converted) == 2 and converted[0]["ad"] != converted[1]["ad"]:
            raise KycError(
                "date_mismatch",
                "BS and AD birth dates do not describe the same day.",
                422,
            )
        if converted:
            values.update(dob_bs=converted[0]["bs"], dob_ad=converted[0]["ad"])
        elif any(k in values for k in ("dob_bs", "dob_ad")):
            values.update(dob_bs="", dob_ad="")
        if values.get("issue_date"):
            if convert_date(values["issue_date"])["warning"]:
                raise KycError(
                    "invalid_date",
                    "Use YYYY-MM-DD BS or YYYY-MM-DD AD for issue date.",
                    422,
                )
        event = {
            "document_id": document_id,
            "corrections": values,
            "reason": payload.reason,
            "at": time.time(),
            "verified": False,
        }
        doc.setdefault("applicant_values", {}).update(values)
        doc.setdefault("correction_history", []).append(event)
        case.setdefault("applicant_corrections", []).append(event)
        p.challenges.pop(case_id, None)
        case["liveness"] = None
        case["consistency"] = check_case(case, p.settings)
        p.store.save(case)
        return public_case(case)
