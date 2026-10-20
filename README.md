# VisionKYC v2 — bilingual document and live video evidence

Sunway College, Kathmandu · Prasanna Sen Thakuri · Shovit Kafle · Samyog Rajbhandari

This revision replaces the original demo descriptor with actual neural inference, adds bilingual OCR and learned field extraction, makes liveness evidence visible, and includes a reviewer workspace and reproducible evaluation tools.

**Readiness:** the software and pretrained inference paths are implemented. This ZIP does **not** contain a completed 50-person study, consented Nepali ID training set, a fine-tuned LiLT checkpoint, or measured presentation-attack resistance. The bundled CRF is a real trained model using **synthetic bilingual seed text**, clearly labelled in the UI. Real dataset collection, fine-tuning, calibration and empirical failure analysis remain required for resubmission. See [requirement status](docs/REQUIREMENTS.md).

## Run with Docker

Prerequisites: Docker Engine + Compose v2 (Linux), or Docker Desktop (Windows/macOS). Allocate about 6 GB RAM for CPU inference; model training may require more. First build needs internet and several GB of free disk space.

From the extracted `kyc-framework` folder:

```bash
docker compose up --build
```

Open **http://localhost:8080**. First start runs `model-setup`, downloading approximately **506 MiB** of pretrained assets and checking their loadability. Assets persist in the `models` volume. The ZIP includes the small trained CRF; the large pretrained assets are downloaded, not bundled. Setup failures stop startup with the exact error; no demo fallback runs.

Get your automatically generated reviewer key in a second terminal:

```bash
docker compose exec backend cat /data/reviewer-key
```

Paste it into the **Reviewer** tab. Optionally copy `.env.example` to `.env` and configure `KYC_REVIEWER_KEY` yourself (at least 24 characters). Do not publish this key.

```bash
# Logs — intentionally omit document contents, tokens and face embeddings
docker compose logs -f model-setup backend frontend

# Stop without deleting model/evidence volumes
docker compose down

# Check actual inference readiness
docker compose exec backend python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health/ready').read().decode())"
```

The port is bound to localhost. Camera access needs localhost or HTTPS. Keep the application local for the college demo; remote deployment requires proper authentication, HTTPS and data governance.

## What is implemented

| Component | Implementation |
|---|---|
| Bilingual OCR | Tesseract `nep+eng`; explicit EasyOCR `ne,en` alternative and comparison CLI |
| NLP | Trained BIO linear-chain CRF with text and layout features; LiLT/XLM-R token-classification training and inference support |
| Document understanding | Bilingual heading classification, per-type required fields, format checks, passport TD3 MRZ checksums, sample markers |
| Portrait extraction | MTCNN or YuNet; exactly one portrait, visible boxes and crop; ambiguous candidates are not silently selected |
| Face recognition | InceptionResnetV1 FaceNet, VGGFace2 and CASIA-WebFace pretrained weights; 512-dimensional embeddings |
| Live video | Browser MediaDevices frames, server-side randomized challenge order, sequence and duplicate-frame checks |
| Liveness | MediaPipe 478 landmarks; eye-close/open and left/right head-turn sequence, continuous FaceNet identity check |
| Consistency | Front/back, primary/secondary identity evidence and user-entered field comparisons; explicit mismatch explanations |
| Review | Protected dashboard, original extracted fields/scores, corrections, approve/reject, audit trail and deletion |
| Evaluation | CER/WER, per-field P/R/F1, detector IoU metrics, ROC/AUC/FAR/FRR, lighting groups, APCER/BPCER, PNG/JSON charts |
| Model failure | `503 model_not_loaded`; visible readiness states; no synthetic embeddings or liveness bypass |

## Demo sequence

1. Open **Demo guide** and inspect the crop diagrams.
2. Create a session with the claimed identity. Enter names in the same script as the ID; append `BS` or `AD` to dates.
3. Upload a clear upright image. Choose **Flat scan** for an already-cropped scan, or **Photograph** to require a detected four-corner boundary. Citizenship requires both sides. A secondary ID is optional.
4. Inspect original, corrected, OCR-box and portrait views. Review extracted values and confidence scores before proceeding.
5. Start the camera session. Close both eyes for approximately two seconds when requested. Follow left/right prompts. The large image is the **last analyzed frame**, with its actual server-produced landmark coordinates, not a decorative overlay on a newer video frame.
6. The final accepted live frame is matched to the document; there is no separate post-challenge selfie upload.
7. Inspect the decision and use the Reviewer tab to correct or decide with a recorded reason.

`datasets/examples/sample-*-INVALID.png` are deliberately invalid synthetic layouts using a public-domain test portrait. They demonstrate OCR and cropping, and **must trigger a sample-document rejection issue**. They are not a real KYC dataset or acceptable identity evidence.

## Decision labels

- `REJECTED_CHECKS`: explicit contradictory/invalid evidence, such as an identified sample document, a confident field mismatch or a calibrated face mismatch.
- `REVIEW_REQUIRED`: missing/uncertain fields, uncalibrated face threshold, or synthetic-seed NER provenance.
- `PASSED_CHECKS`: all configured evidence checks pass, using a real-data NER checkpoint and matching validation calibration.

A reviewer disposition is separate from the immutable automated result. **None of these labels means that a government issuer confirmed document authenticity.** A coherent forgery can pass structural checks. Active facial movement also does not establish certified anti-spoofing.

## Evaluation and training

Start with [DATASET.md](docs/DATASET.md), then [EVALUATION.md](docs/EVALUATION.md). Charts already included in `evaluations/seed_ner/` are measured only on held-out **synthetic text fixtures**. The Evaluation tab labels them accordingly; unmeasured real-data panels stay empty.

```bash
docker compose run --rm tools python -m pytest -q

docker compose run --rm tools python -m app.evaluation.run fields --manifest data/seed.jsonl --engine crf --output /evaluations/seed_ner
```

See [TEST_REPORT.md](docs/TEST_REPORT.md) for actual validation performed and remaining gaps. See [PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for the report draft, architecture, study design and limits. No fabricated team commits or benchmark results are included.
