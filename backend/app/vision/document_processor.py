import re
import uuid
import cv2
from app.vision.image_io import resize_max, preview_url
from app.nlp.validation import classify, validate_mrz, issue, compare_value
from app.nlp.fields import normalize
from app.vision.preprocessing import prepare
from app.nlp.structured import merge_fields, structure, translate_document
from app.nlp.nepali import normalize_number


class DocumentProcessor:
    def __init__(self, settings, registry):
        self.settings = settings
        self.registry = registry

    def process(self, image, requested, side, evidence, mode, cloud_consent=False):
        faces, ocr, ner = self.registry.require("face", "ocr", "ner")
        original = resize_max(image, 2000)
        corrected, preprocessing = prepare(original, mode, ocr.local)
        boundary = preprocessing["boundary"]
        found = faces.detect(corrected)
        text_image = corrected.copy()
        for candidate in found:
            x1, y1, x2, y2 = candidate["box"]
            text_image[y1:y2, x1:x2] = 255
        text, local_results, selected_image = ocr.read_document(
            text_image, cloud_consent
        )
        fields = merge_fields(
            text,
            local_results,
            ner,
            corrected.shape[1],
            corrected.shape[0],
            requested,
            self.settings.min_field_confidence,
        )
        classification = classify(text["text"])
        text["translated_text"] = translate_document(text["text"])
        structured = structure(fields, classification)
        issues = [issue("preprocessing_review", w) for w in preprocessing["warnings"]]
        if not local_results:
            issues.append(
                issue(
                    "cloud_not_cross_checked",
                    "Local OCR did not cross-check the cloud output.",
                )
            )
        if (
            text.get("cloud")
            and text["cloud"]["document_type"] != classification["predicted"]
        ):
            issues.append(
                issue(
                    "header_disagreement",
                    "Cloud type and parsed heading disagree. Inspect the original title.",
                )
            )
        if classification["predicted"] == "unknown":
            # A back side may lack a heading; it must be linked to a recognized front.
            issues.append(
                issue(
                    "unclassified_side",
                    "No unambiguous document heading detected on this side.",
                    "review" if side == "back" else "reject",
                )
            )
        elif classification["predicted"] != requested:
            issues.append(
                issue(
                    "wrong_document_type",
                    f"Detected {classification['predicted']}; expected {requested}.",
                    "reject",
                )
            )
        fake = re.search(
            r"\b(sample|specimen|fake|dummy|not valid|test document)\b|नमूना|नक्कली",
            normalize("\n".join([text["text"]] + [v["text"] for v in local_results])),
        )
        if fake:
            issues.append(
                issue(
                    "sample_document",
                    "A sample/fake/specimen marker was detected.",
                    "reject",
                )
            )
        if text["confidence"] < self.settings.min_ocr_confidence:
            issues.append(issue("low_ocr", "OCR confidence is low."))
        for field, data in fields.items():
            if data["conflicts"]:
                issues.append(
                    issue(
                        "engine_disagreement",
                        f"OCR candidates disagree on {field}; inspect alternatives.",
                        field=field,
                    )
                )
            selected = data["selected"]
            if selected and selected.get("needs_review"):
                issues.append(
                    issue(
                        "low_field_confidence",
                        f"Confirm the extracted {field}.",
                        field=field,
                    )
                )
            if selected and selected.get("date", {}).get("warning"):
                issues.append(
                    issue("date_review", selected["date"]["warning"], field=field)
                )
        number = fields["document_number"]["selected"]
        if number and not normalize_number(number["value"], requested)["valid"]:
            issues.append(
                issue(
                    "id_format",
                    f"{requested} number does not match the configurable format rules.",
                )
            )
        mrz = validate_mrz(text["text"]) if requested == "passport" else None
        if requested == "passport" and side == "front":
            if not mrz["present"]:
                issues.append(
                    issue(
                        "missing_mrz",
                        "Passport MRZ was not read; inspect the two machine-readable lines.",
                    )
                )
            elif not mrz["valid"]:
                issues.append(
                    issue("mrz_checksum", "Passport MRZ checksum failure.", "reject")
                )
            elif (
                number
                and compare_value(
                    "document_number", number["value"], mrz["document_number"]
                )
                == "mismatch"
            ):
                issues.append(
                    issue(
                        "mrz_id_mismatch",
                        "Printed document number and MRZ differ.",
                        "reject",
                    )
                )
        annotated = corrected.copy()
        for token in text["tokens"]:
            x1, y1, x2, y2 = token["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (225, 180, 45), 1)
        for index, face in enumerate(found):
            x1, y1, x2, y2 = face["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (70, 200, 50), 3)
            cv2.putText(
                annotated,
                f"Face {index + 1}: {face['confidence']:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 120, 20),
                2,
            )
        crop = None
        embedding = None
        if side == "front":
            if len(found) != 1:
                issues.append(
                    issue(
                        "portrait_count",
                        f"Detected {len(found)} portrait candidates. Exactly one is required; no automatic largest-face selection.",
                        "reject",
                    )
                )
            else:
                crop = faces.crop(corrected, found[0])
                embedding = faces.embed(crop).tolist()
                if min(crop.shape[:2]) / preprocessing["upscale_factor"] < 80:
                    issues.append(
                        issue(
                            "small_portrait",
                            "Portrait has fewer than 80 pixels on one side.",
                        )
                    )
        return {
            "id": str(uuid.uuid4()),
            "evidence": evidence,
            "side": side,
            "requested_type": requested,
            "input_mode": mode,
            "classification": classification,
            "ocr": text,
            "fields": fields,
            "structured": structured,
            "preprocessing": preprocessing,
            "ner": self.settings.ner_engine,
            "ner_provenance": ner.provenance,
            "boundary": boundary,
            "portrait_candidates": found,
            "mrz": mrz,
            "issues": issues,
            "previews": {
                "original": preview_url(original),
                "ocr_selected": preview_url(selected_image),
                "rectified": preview_url(corrected),
                "annotated": preview_url(annotated),
                "portrait": preview_url(crop, 256) if crop is not None else None,
            },
            "_embedding": embedding,
            "authenticity": "NOT_VERIFIED_WITH_ISSUER",
        }
