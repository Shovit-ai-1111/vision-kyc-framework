# Validation performed for this ZIP

## Passed locally

- Backend: 32 pytest checks (authorization, consent, explicit missing models, reviewer revision protection, field comparisons, MRZ, metrics, learned CRF and liveness transitions).
- Frontend: 3 Vitest component checks; Vite production build succeeded.
- Python compile checks and Ruff undefined-name/import checks.
- Real-model API smoke: document analysis succeeded, MediaPipe returned 478 points, duplicate frame returned 409 `replayed_frame`, finish after failed liveness was blocked, and reviewer received FAILED without embeddings.
- Compose YAML parsed and dependency/volume wiring was inspected; this is not a container execution test.
- Actual downloaded FaceNet VGGFace2 and CASIA-WebFace inference, each with MTCNN and YuNet: 512-dimensional normalized embeddings and a detected public test portrait.
- Actual MediaPipe inference returned 478 landmarks.
- Tesseract with Nepali/English language assets and EasyOCR with Nepali/English weights ran actual text inference.
- All three intentionally invalid synthetic layouts yielded one portrait, correct heading family, six extracted fields, and explicit sample-document rejection issues.
- CRF training and held-out synthetic field metrics generated the shipped JSON and PNG charts.
- Python runtime dependency resolution completed successfully with the specified top-level requirements. facenet-pytorch legacy metadata is intentionally bypassed as explained in MODELS.md.

Machine-readable evidence: `model-smoke-results.json`, `document-smoke-results.json`, `api-smoke-results.json`, and `evaluations/seed_ner/`. Smoke results are operational checks, not accuracy benchmarks. No actual user IDs or session secrets are included.

## Not verified here

- Full Docker image build / Compose startup: no Docker engine was available.
- Physical live-camera completion, low-light robustness, print/phone/replay attacks: require real devices and participants.
- Full browser screenshot/end-to-end UI verification: the browser binary download failed in this environment. Production compilation and component tests passed.
- Training/inference of a fine-tuned LiLT/XLM-R KYC checkpoint: no consented annotated KYC dataset was supplied.
- Fifty-person ROC evaluation, bilingual real-document CER, and twenty empirical failure cases: no such dataset was supplied.

Local testing used Python 3.12; Docker targets Python 3.11 with explicit dependencies. The host/container version difference is another reason a Docker smoke run is still required on the target machine.
