# Docker and operations

Run `docker compose up --build` from the project root. No environment file is required for defaults. Model setup runs first; backend waits for successful completion; nginx starts once backend process health passes. The UI then independently checks model readiness. Inference remains unavailable if any selected component fails loading.

Services: `model-setup`, `backend`, `frontend`, and an opt-in `tools` profile. Named volumes retain models and SQLite cases; `datasets/` and `evaluations/` are bind-mounted for experiments. The backend runs as UID 10001 and has no published host port. Tools runs as root inside its container so it can write experiment outputs on different host platforms. Files produced by tools may be root-owned on Linux; adjust ownership on your host if necessary.

The generated reviewer key is stored at `/data/reviewer-key`. To use your own key, set `KYC_REVIEWER_KEY` in `.env`. If you set an environment key, the file's generated key is not the active key.

```bash
docker compose logs -f model-setup
docker compose logs -f backend frontend
docker compose exec backend cat /data/reviewer-key
docker compose run --rm tools python -m pytest -q
```

Change model selection in `.env` and recreate the backend. Examples: `KYC_OCR_ENGINE=easyocr`, `KYC_DETECTOR=yunet`, `KYC_FACE_WEIGHTS=casia-webface`. Every model choice is explicit. A calibration must match the selected detector and weight checksum. For `KYC_NER_ENGINE=transformer`, first train a checkpoint to `/models/ner-transformer`; otherwise readiness fails.

```bash
docker compose up -d --force-recreate backend
```

If a download fails, inspect the setup logs and retry `docker compose run --rm model-setup`. Completed assets persist. No substitute source/model is attempted. For a checksum mismatch, inspect the asset and manifest; do not disable verification casually. If you deliberately want to inspect the unavailable-model UI before setup, run `docker compose up --build --no-deps backend frontend`; inference will return 503.

Default retention is 24 hours, checked periodically and during reads. Deleting cases removes database records but does not guarantee forensic erasure of SQLite pages or backups. Disk/volume encryption and managed retention are needed for sensitive real deployments. Live frames are processed transiently; selected document/face previews, OCR data and review records are retained. No full video recording is stored. Consent is required before creating a case.

One backend worker is intentional: challenge state is in memory. Restarting invalidates active challenges. The API uses a shared demo reviewer key, not a production multi-user identity provider. Do not expose the service publicly as-is. The Docker engine is unavailable in the delivery environment, so a complete container build/start remains a host-side validation step; local model inference and application checks are recorded separately.
