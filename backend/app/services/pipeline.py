import copy
import time
from threading import RLock
import numpy as np
from app.core.errors import KycError
from app.nlp.validation import check_case, issue
from app.vision.document_processor import DocumentProcessor
from app.vision.image_io import decode_image, decode_base64, preview_url
from app.vision.liveness import Challenge
from app.vision.guidance import face_guidance, no_face_guidance
from app.services.decision import decide
from app.services.store import Store


def observe(challenge, image, faces, landmarks, now):
    metrics = landmarks.analyze(image)
    crop, detection = faces.one(image)
    x1, y1, x2, y2 = detection["box"]
    import cv2

    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    metrics["brightness"] = float(gray.mean())
    metrics["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    box = [
        x1 / image.shape[1],
        y1 / image.shape[0],
        x2 / image.shape[1],
        y2 / image.shape[0],
    ]
    metrics["guidance"] = face_guidance(
        box,
        metrics["brightness"],
        metrics["sharpness"],
        challenge.steps[challenge.index],
        metrics["nose_offset"],
    )
    embedding = faces.embed(crop)
    if challenge.reference is None:
        challenge.reference = embedding
    similarity = faces.similarity(challenge.reference, embedding)
    metrics["continuity_similarity"] = similarity
    challenge.continuity_min = min(challenge.continuity_min, similarity)
    if similarity < challenge.settings.continuity_threshold:
        challenge.status = "FAILED"
        challenge.issue = "Face continuity was lost. Restart the camera session."
        challenge.last = metrics
        return
    if not metrics["guidance"]["ready"]:
        challenge.last = metrics
        challenge.reset_partial()
        challenge.issue = metrics["guidance"]["prompt"]
        return
    challenge.advance(metrics, now)
    # Final selfie comes from the last accepted challenge frame, never a later upload.
    if challenge.status == "PASSED":
        challenge.best_embedding = embedding
        challenge.selfie = preview_url(crop, 400)


def public_case(case):
    case = copy.deepcopy(case)
    for document in case["documents"]:
        document.pop("_embedding", None)
    return case


class Pipeline:
    def __init__(self, settings, registry):
        self.settings = settings
        self.registry = registry
        self.store = Store(settings)
        self.documents = DocumentProcessor(settings, registry)
        self.lock = RLock()
        self.challenges = {}
        self.attempts = {}

    def cleanup(self):
        now = time.monotonic()
        for key, challenge in list(self.challenges.items()):
            if now - challenge.started > self.settings.challenge_ttl_seconds + 30:
                self.challenges.pop(key, None)
                if challenge.status == "IN_PROGRESS":
                    try:
                        case = self.store.get(key, reviewer=True)
                        case["liveness"] = {
                            "status": "FAILED",
                            "issue": "Challenge timed out or camera session was abandoned.",
                            "events": challenge.events,
                            "steps": challenge.steps,
                            "frames": challenge.frames,
                        }
                        self.store.save(case)
                    except KycError as error:
                        if error.status_code != 404:
                            raise
        self.store.purge()

    def add_document(self, case_id, token, data, requested, side, evidence, mode):
        with self.lock:
            case = self.store.get(case_id, token)
            if case["result"]:
                raise KycError(
                    "case_complete",
                    "Completed cases are immutable. Start a new case.",
                    409,
                )
            if evidence == "primary" and requested != case["document_type"]:
                raise KycError(
                    "type_mismatch",
                    "Primary evidence must use the selected case document type.",
                )
            duplicate = next(
                (
                    d
                    for d in case["documents"]
                    if d["evidence"] == evidence and d["side"] == side
                ),
                None,
            )
            if duplicate:
                raise KycError(
                    "duplicate_side",
                    "This evidence side already exists. Remove it before uploading again.",
                    409,
                )
            if len(case["documents"]) >= 4:
                raise KycError("document_limit", "Maximum four sides per case.")
            revision = case["revision"]
            cloud_consent = case.get("cloud_ocr_consent", False)
        # Slow OCR/provider retries must not block the shared camera lock.
        document = self.documents.process(
            decode_image(data, self.settings),
            requested,
            side,
            evidence,
            mode,
            cloud_consent,
        )
        with self.lock:
            case = self.store.get(case_id, token)
            if case["result"] or case["revision"] != revision:
                raise KycError(
                    "revision_conflict",
                    "Case changed during analysis. Reload and retry.",
                    409,
                )
            case["documents"].append(document)
            case["consistency"] = check_case(case, self.settings)
            self.challenges.pop(case_id, None)
            case["liveness"] = None
            self.store.save(case)
            return public_case(case)

    def start(self, case_id, token):
        with self.lock:
            self.registry.require("face", "landmarks")
            self.cleanup()
            case = self.store.get(case_id, token)
            if case["result"]:
                raise KycError("case_complete", "Case already completed.", 409)
            primary = next(
                (
                    d
                    for d in case["documents"]
                    if d["evidence"] == "primary" and d["side"] == "front"
                ),
                None,
            )
            if not primary or primary["_embedding"] is None:
                raise KycError(
                    "portrait_required",
                    "Upload a primary document with exactly one portrait before the live session.",
                    409,
                )
            if (
                len(self.challenges) >= self.settings.max_active_challenges
                and case_id not in self.challenges
            ):
                raise KycError(
                    "camera_capacity",
                    "Too many active camera sessions. Retry later.",
                    503,
                )
            count = case.get("challenge_attempts", 0)
            if count >= 5:
                raise KycError(
                    "attempt_limit", "Five camera attempts used. Start a new case.", 429
                )
            case["challenge_attempts"] = count + 1
            challenge = Challenge(self.settings)
            self.challenges[case_id] = challenge
            case["liveness"] = {"status": "IN_PROGRESS", "challenge_id": challenge.id}
            self.store.save(case)
            return challenge.response()

    def frame(self, case_id, token, challenge_id, payload):
        with self.lock:
            case = self.store.get(case_id, token)
            challenge = self.challenges.get(case_id)
            if not challenge or challenge.id != challenge_id:
                raise KycError(
                    "invalid_challenge",
                    "Challenge expired or server restarted. Start a new challenge.",
                    409,
                )
            faces, landmarks = self.registry.require("face", "landmarks")
            image = decode_base64(payload.image, self.settings)
            now = time.monotonic()
            try:
                challenge.accept_frame(image, payload.sequence, now)
                observe(challenge, image, faces, landmarks, time.monotonic())
            except KycError as error:
                if error.status_code in (503, 409, 410, 429):
                    if challenge.status == "FAILED":
                        case["liveness"] = {
                            "status": "FAILED",
                            "issue": challenge.issue,
                            "events": challenge.events,
                        }
                        self.store.save(case)
                    raise
                challenge.reset_partial()
                import re

                counts = re.findall(r"(?:detected|Detected) (\d+)", error.message)
                count = int(counts[0]) if counts else 0
                challenge.last = {
                    "guidance": no_face_guidance(error.message, count),
                    "landmarks": [],
                }
                challenge.issue = error.message
            if challenge.status in ("PASSED", "FAILED"):
                case["liveness"] = {
                    "status": challenge.status,
                    "challenge_id": challenge.id,
                    "events": challenge.events,
                    "steps": challenge.steps,
                    "frames": challenge.frames,
                    "duration_seconds": round(time.monotonic() - challenge.started, 2),
                    "continuity_min": challenge.continuity_min,
                    "baseline_ear": challenge.baseline,
                    "issue": challenge.issue,
                    "assurance": "active challenge; not certified PAD",
                }
                if challenge.selfie:
                    case["selfie_preview"] = challenge.selfie
                self.store.save(case)
            return challenge.response()

    def finish(self, case_id, token):
        with self.lock:
            case = self.store.get(case_id, token)
            if case["result"]:
                return public_case(case)
            ch = self.challenges.get(case_id)
            if (
                not ch
                or ch.status != "PASSED"
                or ch.best_embedding is None
                or ch.passed_at is None
                or time.monotonic() - ch.passed_at > 30
            ):
                raise KycError(
                    "live_evidence_required",
                    "Complete a fresh active camera challenge first.",
                    409,
                )
            (faces,) = self.registry.require("face")
            primary = next(
                d
                for d in case["documents"]
                if d["evidence"] == "primary" and d["side"] == "front"
            )
            score = faces.similarity(np.array(primary["_embedding"]), ch.best_embedding)
            consistency = check_case(case, self.settings)
            threshold = (
                self.registry.calibration["threshold"]
                if self.registry.calibration
                else self.settings.similarity_threshold
            )
            for document in case["documents"]:
                if document["evidence"] == "secondary" and document["_embedding"]:
                    secondary = faces.similarity(
                        np.array(document["_embedding"]), ch.best_embedding
                    )
                    consistency["comparisons"].append(
                        {
                            "field": "face",
                            "source_a": document["id"],
                            "source_b": "live_selfie",
                            "value_a": secondary,
                            "value_b": threshold,
                            "result": "match" if secondary >= threshold else "mismatch",
                        }
                    )
                    if secondary < threshold:
                        consistency["issues"].append(
                            issue(
                                "secondary_face_mismatch",
                                "Secondary ID portrait differs from the live selfie.",
                                "reject" if self.registry.calibration else "review",
                            )
                        )
            case["consistency"] = consistency
            case["result"] = decide(case, self.registry, consistency, score)
            for document in case["documents"]:
                document["_embedding"] = None
            self.store.save(case)
            self.challenges.pop(case_id, None)
            return public_case(case)
