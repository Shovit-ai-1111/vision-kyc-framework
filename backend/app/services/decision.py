from app.nlp.validation import issue


def decide(case, registry, consistency, similarity):
    reasons = list(consistency["issues"])
    live = case.get("liveness")
    if not live or live["status"] != "PASSED":
        reasons.append(
            issue(
                "liveness_failed", "Active liveness challenge did not pass.", "reject"
            )
        )
    calibrated = registry.calibration is not None
    threshold = (
        registry.calibration["threshold"]
        if calibrated
        else registry.settings.similarity_threshold
    )
    if similarity is None:
        reasons.append(
            issue(
                "missing_face_score",
                "Face verification evidence is unavailable.",
                "reject",
            )
        )
    elif similarity < threshold:
        reasons.append(
            issue(
                "face_mismatch",
                f"Cosine similarity {similarity:.4f} is below {threshold:.4f}.",
                "reject" if calibrated else "review",
            )
        )
    if not calibrated:
        reasons.append(
            issue(
                "uncalibrated_face",
                "Collect validation pairs and install a matching calibration.json before automated pass.",
            )
        )
    provenance = registry.status()["ner_provenance"] or {}
    if provenance.get("data_kind") != "consented_real":
        reasons.append(
            issue(
                "seed_ner",
                "NER was trained on synthetic seed text; real KYC validation is still required.",
            )
        )
    if any(r["severity"] == "reject" for r in reasons):
        status = "REJECTED_CHECKS"
    elif reasons:
        status = "REVIEW_REQUIRED"
    else:
        status = "PASSED_CHECKS"
    return {
        "status": status,
        "reasons": reasons,
        "face": {
            "cosine_similarity": similarity,
            "threshold": threshold,
            "calibrated": calibrated,
            "weights": registry.settings.face_weights,
            "detector": registry.settings.detector,
            "score_is_probability": False,
        },
        "authenticity": "NOT_VERIFIED_WITH_ISSUER",
        "note": "This academic result covers submitted evidence checks, not government issuance/authenticity or certified presentation-attack detection.",
    }
