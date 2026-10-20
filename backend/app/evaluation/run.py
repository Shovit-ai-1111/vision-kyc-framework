"""Reproducible metrics and charts. No populated results without observations."""

import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from app.core.config import Settings
from app.models.schemas import FIELDS
from app.evaluation.dataset import load_jsonl, validate_documents
from app.evaluation.metrics import ocr_counts, prf, binary_metrics, iou
from app.nlp.fields import normalize


def load_rows(path):
    p = Path(path)
    return list(csv.DictReader(p.open())) if p.suffix == ".csv" else load_jsonl(p)


def resolve(manifest, value):
    return Path(manifest).resolve().parent / value


def write(args, summary, rows):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary.update(
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        generated_at=time.time(),
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    )
    (output / "observations.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows)
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i, chart in enumerate(summary.get("charts", [])):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#f8fbfa")
        ax.set_facecolor("#f8fbfa")
        if chart["type"] == "bar":
            ax.barh(
                [d["label"] for d in chart["data"]],
                [d["value"] for d in chart["data"]],
                color="#078375",
            )
            ax.invert_yaxis()
            ax.set_xlim(0, chart.get("max", 1))
            ax.set_xlabel("Ratio" if chart.get("unit") != "number" else "Value")
        else:
            for s in chart["series"]:
                ax.plot(
                    [p[0] for p in s["points"]],
                    [p[1] for p in s["points"]],
                    label=s["label"],
                )
            ax.set_xlabel(chart.get("xLabel", ""))
            ax.set_ylabel(chart.get("yLabel", ""))
            ax.legend(loc="best")
        ax.set_title(chart["title"])
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output / f"chart-{i + 1}.png", dpi=160)
        plt.close(fig)
    print(
        f"Wrote {output}/summary.json and {len(summary.get('charts', []))} measured charts"
    )


def baseline_fields(text):
    """Explicitly evaluated label/colon baseline; never used by online inference."""
    import re

    patterns = {
        "name": r"^(?:full name|name|नाम थर)\s*:",
        "father_name": r"^(?:father name|father's name|बाबुको नाम)\s*:",
        "dob": r"^(?:date of birth|dob|जन्म मिति)\s*:",
        "document_number": r"^(?:document number|pan number|passport number|citizenship no|नागरिकता नं)\s*:",
        "district": r"^(?:district|जिल्ला)\s*:",
        "issue_date": r"^(?:issue date|जारी मिति)\s*:",
    }
    values = {}
    for line in text.splitlines():
        for field, pattern in patterns.items():
            match = re.match(pattern, line.strip(), re.I)
            if match:
                values[field] = line[match.end() :].strip()
    return values


def fields(args):
    from app.nlp.fields import Extractor

    rows = load_jsonl(args.manifest)
    counts = validate_documents(rows)
    selected = [r for r in rows if r["split"] == args.split]
    if not selected:
        raise ValueError("No evaluation rows in selected split")
    settings = Settings(ner_engine=args.engine)
    extractor = Extractor(settings)
    results = []
    metrics = {}
    charts = []
    for model in ("label_regex_baseline", args.engine):
        totals = {f: [0, 0, 0] for f in FIELDS}
        for r in selected:
            if model == "label_regex_baseline":
                pred = baseline_fields(r["text"])
            else:
                pred = {
                    f: v["selected"]["value"]
                    for f, v in extractor.extract(
                        r["tokens"], r["width"], r["height"]
                    ).items()
                    if v["selected"]
                }
            for f in FIELDS:
                truth = normalize(r["fields"].get(f, ""))
                guess = normalize(pred.get(f, ""))
                if truth and guess == truth:
                    totals[f][0] += 1
                else:
                    if guess:
                        totals[f][1] += 1
                    if truth:
                        totals[f][2] += 1
            results.append(
                {
                    "id": r["id"],
                    "model": model,
                    "truth": r["fields"],
                    "prediction": pred,
                }
            )
        per_field = {f: prf(*values) for f, values in totals.items()}
        metrics[model] = per_field
        charts.append(
            {
                "type": "bar",
                "title": f"{model}: exact field F1 ({args.split})",
                "data": [
                    {"label": f.replace("_", " "), "value": v["f1"]}
                    for f, v in per_field.items()
                ],
            }
        )
    write(
        args,
        {
            "kind": "fields",
            "data_kind": rows[0]["data_kind"],
            "counts": counts,
            "metrics": metrics,
            "charts": charts,
            "notes": "Exact field match after NFC, digit and whitespace normalization. OCR-token annotations supplied; this measures NER, not end-to-end OCR. Synthetic seed metrics do not represent real IDs."
            if rows[0]["data_kind"] == "synthetic_seed"
            else "Held-out NER on supplied token annotations. End-to-end OCR+NER must also be reported.",
        },
        results,
    )


def ocr(args):
    from app.vision.ocr import OCR
    from app.vision.image_io import decode_image

    rows = load_jsonl(args.manifest)
    validate_documents(rows, require_images=True)
    rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise ValueError("No evaluation documents")
    settings = Settings()
    results = []
    metrics = {}
    for engine in ("tesseract", "easyocr"):
        reader = OCR(settings, engine)
        totals = dict(character_errors=0, characters=0, word_errors=0, words=0)
        seconds = []
        for r in rows:
            image = decode_image(
                resolve(args.manifest, r["image"]).read_bytes(), settings
            )
            start = time.perf_counter()
            prediction = reader.read(image)
            seconds.append(time.perf_counter() - start)
            values = ocr_counts(r["text"], prediction["text"])
            if not values["characters"]:
                raise ValueError("Gold transcription may not be blank")
            for k, v in values.items():
                totals[k] += v
            results.append(
                {
                    "id": r["id"],
                    "engine": engine,
                    **values,
                    "prediction": prediction["text"],
                    "seconds": seconds[-1],
                }
            )
        metrics[engine] = {
            "cer": totals["character_errors"] / totals["characters"],
            "wer": totals["word_errors"] / totals["words"],
            "mean_seconds": float(np.mean(seconds)),
            **totals,
        }
    max_error = max(1, max(v["cer"] for v in metrics.values()))
    write(
        args,
        {
            "kind": "ocr",
            "data_kind": rows[0]["data_kind"],
            "counts": {"documents": len(rows)},
            "metrics": metrics,
            "charts": [
                {
                    "type": "bar",
                    "title": "OCR character error rate — lower is better",
                    "max": max_error,
                    "data": [
                        {"label": e, "value": v["cer"]} for e, v in metrics.items()
                    ],
                }
            ],
            "notes": "Micro CER counts Unicode code points (not graphemes); NFC and whitespace normalization only. CER can exceed 100% due to insertions.",
        },
        results,
    )


def face(args):
    from sklearn.metrics import roc_curve, roc_auc_score
    from app.vision.face_verifier import FaceEngine
    from app.vision.image_io import decode_image

    rows = load_rows(args.manifest)
    subjects = {}
    images = {}
    if not rows:
        raise ValueError("No face pairs")
    for r in rows:
        if r["split"] not in ("validation", "test"):
            raise ValueError("Face pairs require validation/test split")
        if not r.get("consent_id"):
            raise ValueError("Consent reference required for each pair")
        for key in ("subject_a", "subject_b"):
            if r[key] in subjects and subjects[r[key]] != r["split"]:
                raise ValueError("Subject leakage between validation and test")
            subjects[r[key]] = r["split"]
        for key in ("image_a", "image_b"):
            if r[key] in images and images[r[key]] != r["split"]:
                raise ValueError("Image leakage between splits")
            images[r[key]] = r["split"]
        if r["image_a"] == r["image_b"]:
            raise ValueError("An image cannot be compared with itself")
        r["genuine"] = r["subject_a"] == r["subject_b"]
    if len(subjects) < 50 and not args.allow_small:
        raise ValueError(
            "At least 50 people required; use --allow-small only for an explicitly labelled pilot"
        )
    summary = {
        "kind": "face",
        "data_kind": "consented_real" if len(subjects) >= 50 else "small_pilot",
        "counts": {"people": len(subjects), "pairs": len(rows)},
        "metrics": {},
        "charts": [],
        "notes": "Thresholds selected ONLY on validation identities; final metrics use disjoint test identities. Detection failures count as rejected matches. Model scores are not probabilities.",
    }
    observations = []
    for detector in args.detectors.split(","):
        for weights in ("vggface2", "casia-webface"):
            settings = Settings(detector=detector, face_weights=weights)
            engine = FaceEngine(settings)
            cache = {}
            scored = []
            name = f"{detector}/{weights}"
            for row in rows:
                errors = []
                embeddings = []
                for field in ("image_a", "image_b"):
                    path = str(resolve(args.manifest, row[field]))
                    if path not in cache:
                        try:
                            image = decode_image(Path(path).read_bytes(), settings)
                            crop, _ = engine.one(image)
                            cache[path] = (engine.embed(crop), None)
                        except Exception as error:
                            from app.core.errors import KycError

                            if not isinstance(error, KycError):
                                raise
                            cache[path] = (None, error.code)
                    vector, error = cache[path]
                    embeddings.append(vector)
                    if error:
                        errors.append(error)
                score = engine.similarity(*embeddings) if not errors else None
                scored.append({**row, "score": score, "errors": errors, "model": name})
            val = [
                r
                for r in scored
                if r["split"] == "validation" and r["score"] is not None
            ]
            test = [r for r in scored if r["split"] == "test"]
            if (
                len({r["genuine"] for r in val}) < 2
                or len({r["genuine"] for r in test}) < 2
            ):
                raise ValueError(
                    "Both genuine and impostor pairs required in validation and test"
                )
            labels = [r["genuine"] for r in val]
            scores = [r["score"] for r in val]
            fpr, tpr, thresholds = roc_curve(labels, scores)
            eligible = [
                i
                for i, t in enumerate(thresholds)
                if np.isfinite(t) and -1 <= t <= 1 and fpr[i] <= args.target_far
            ]
            if not eligible:
                raise ValueError(
                    "No finite threshold meets target FAR; collect more data or change the stated research target"
                )
            selected = max(eligible, key=lambda i: (tpr[i], thresholds[i]))
            threshold = float(thresholds[selected])
            stats = binary_metrics(
                [r["genuine"] for r in test],
                [r["score"] is not None and r["score"] >= threshold for r in test],
            )
            stats.update(
                threshold=threshold,
                validation_auc=float(roc_auc_score(labels, scores)),
                test_detection_failures=sum(r["score"] is None for r in test),
                test_pairs=len(test),
            )
            successful = [r for r in test if r["score"] is not None]
            if len({r["genuine"] for r in successful}) == 2:
                fx, ty, th = roc_curve(
                    [r["genuine"] for r in successful], [r["score"] for r in successful]
                )
                idx = int(np.argmin(abs(fx - (1 - ty))))
                stats["test_auc"] = float(
                    roc_auc_score(
                        [r["genuine"] for r in successful],
                        [r["score"] for r in successful],
                    )
                )
                stats["eer_approx"] = float((fx[idx] + 1 - ty[idx]) / 2)
                summary["charts"].append(
                    {
                        "type": "line",
                        "title": f"{name}: held-out ROC",
                        "xLabel": "False acceptance rate",
                        "yLabel": "True acceptance rate",
                        "series": [
                            {
                                "label": name,
                                "points": np.column_stack([fx, ty]).tolist(),
                            }
                        ],
                    }
                )
            stats["lighting"] = {}
            for lighting in sorted({r.get("lighting", "unspecified") for r in test}):
                group = [
                    r for r in test if r.get("lighting", "unspecified") == lighting
                ]
                stats["lighting"][lighting] = binary_metrics(
                    [r["genuine"] for r in group],
                    [r["score"] is not None and r["score"] >= threshold for r in group],
                )
            stats["per_subject_genuine"] = {}
            for person in sorted({r["subject_a"] for r in test if r["genuine"]}):
                group = [r for r in test if r["genuine"] and r["subject_a"] == person]
                stats["per_subject_genuine"][person] = {
                    "pairs": len(group),
                    "acceptance_rate": sum(
                        r["score"] is not None and r["score"] >= threshold
                        for r in group
                    )
                    / len(group),
                }
            summary["metrics"][name] = stats
            for r in scored:
                r["predicted_same"] = r["score"] is not None and r["score"] >= threshold
            observations.extend(scored)
            val_subjects = {
                r[k]
                for r in rows
                if r["split"] == "validation"
                for k in ("subject_a", "subject_b")
            }
            artifact = {
                "split": "validation",
                "subjects": len(val_subjects),
                "study_subjects": len(subjects),
                "genuine_count": sum(labels),
                "impostor_count": len(labels) - sum(labels),
                "threshold": threshold,
                "target_far": args.target_far,
                "observed_far": float(fpr[selected]),
                "detector": detector,
                "face_weights": weights,
                "sha256": hashlib.sha256(
                    (settings.weights_dir / f"facenet-{weights}.pt").read_bytes()
                ).hexdigest(),
                "preprocessing": "bbox-margin-0.10-resize160-rgb-fixed-standardization",
                "manifest_sha256": hashlib.sha256(
                    Path(args.manifest).read_bytes()
                ).hexdigest(),
            }
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"calibration-{detector}-{weights}.json").write_text(
                json.dumps(artifact, indent=2)
            )
            summary["charts"].append(
                {
                    "type": "bar",
                    "title": f"{name}: test accuracy by lighting",
                    "data": [
                        {"label": light, "value": v["accuracy"]}
                        for light, v in stats["lighting"].items()
                    ],
                }
            )
    write(args, summary, observations)


def detectors(args):
    from app.vision.face_verifier import FaceEngine
    from app.vision.image_io import decode_image

    rows = load_jsonl(args.manifest)
    results = []
    metrics = {}
    if not rows:
        raise ValueError("No detector annotations")
    for name in ("mtcnn", "yunet"):
        settings = Settings(detector=name)
        engine = FaceEngine(settings)
        tp = fp = fn = 0
        times = []
        for r in rows:
            if not r.get("consent_id"):
                raise ValueError("Consent reference required")
            image = decode_image(
                resolve(args.manifest, r["image"]).read_bytes(), settings
            )
            start = time.perf_counter()
            found = engine.detect(image)
            times.append(time.perf_counter() - start)
            expected = list(r["boxes"])
            hits = 0
            for d in sorted(found, key=lambda f: f["confidence"], reverse=True):
                matches = [(iou(d["box"], b), i) for i, b in enumerate(expected)]
                if matches and max(matches)[0] >= 0.5:
                    expected.pop(max(matches)[1])
                    hits += 1
            tp += hits
            fp += len(found) - hits
            fn += len(expected)
            results.append(
                {
                    "id": r["id"],
                    "detector": name,
                    "expected": r["boxes"],
                    "detected": found,
                    "tp": hits,
                    "fp": len(found) - hits,
                    "fn": len(expected),
                    "seconds": times[-1],
                }
            )
        metrics[name] = {**prf(tp, fp, fn), "mean_seconds": float(np.mean(times))}
    write(
        args,
        {
            "kind": "detectors",
            "data_kind": "consented_real",
            "counts": {"images": len(rows)},
            "metrics": metrics,
            "charts": [
                {
                    "type": "bar",
                    "title": "Face detector recall at IoU ≥ 0.5",
                    "data": [
                        {"label": k, "value": v["recall"]} for k, v in metrics.items()
                    ],
                }
            ],
            "notes": "Bounding boxes are manually annotated. Detection and recognition are different tasks; both are evaluated separately.",
        },
        results,
    )


def liveness(args):
    rows = load_jsonl(args.manifest)
    observations = []
    groups = {}
    people = set()
    if not rows:
        raise ValueError("No attack trials")
    for r in rows:
        if not r.get("consent_id"):
            raise ValueError("Consent reference required")
        if r["attack_type"] not in (
            "bona_fide",
            "print",
            "phone_photo",
            "phone_replay",
        ):
            raise ValueError("Unsupported trial type")
        record = json.loads(resolve(args.manifest, r["record"]).read_text())
        evidence = record.get("liveness")
        if not evidence or evidence["status"] not in ("PASSED", "FAILED"):
            raise ValueError(
                "Use completed server observations; timeouts/aborts must be exported as FAILED trials"
            )
        passed = evidence["status"] == "PASSED"
        groups.setdefault(r["attack_type"], []).append(passed)
        people.add(r["subject_id"])
        observations.append(
            {
                "trial": r["id"],
                "subject": r["subject_id"],
                "attack_type": r["attack_type"],
                "passed": passed,
                "events": evidence.get("events", []),
            }
        )
    if not all(
        kind in groups for kind in ("bona_fide", "print", "phone_photo", "phone_replay")
    ):
        raise ValueError(
            "Include bona fide, print, phone-photo and phone-replay trials"
        )
    metrics = {
        "bpcer": 1 - sum(groups["bona_fide"]) / len(groups["bona_fide"]),
        "apcer": {k: sum(v) / len(v) for k, v in groups.items() if k != "bona_fide"},
    }
    write(
        args,
        {
            "kind": "liveness",
            "data_kind": "consented_real",
            "counts": {
                "participants": len(people),
                **{k: len(v) for k, v in groups.items()},
            },
            "metrics": metrics,
            "charts": [
                {
                    "type": "bar",
                    "title": "Attack acceptance and bona fide rejection — lower is better",
                    "data": [
                        {"label": k + " APCER", "value": v}
                        for k, v in metrics["apcer"].items()
                    ]
                    + [{"label": "Bona fide BPCER", "value": metrics["bpcer"]}],
                }
            ],
            "notes": "Measured from saved online randomized challenges, with manually annotated attack types. This is an academic protocol, not PAD certification.",
        },
        observations,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    for name, handler in [
        ("fields", fields),
        ("ocr", ocr),
        ("face", face),
        ("detectors", detectors),
        ("liveness", liveness),
    ]:
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--output", required=True)
        p.set_defaults(handler=handler)
        if name in ("fields", "ocr"):
            p.add_argument("--split", choices=["validation", "test"], default="test")
        if name == "fields":
            p.add_argument("--engine", choices=["crf", "transformer"], default="crf")
        if name == "face":
            p.add_argument("--detectors", default="mtcnn,yunet")
            p.add_argument("--target-far", type=float, default=0.01)
            p.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
