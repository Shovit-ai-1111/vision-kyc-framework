# Reproduce training, metrics and charts

All commands run from the project root after model setup. Output directories are bind-mounted at `evaluations/`. The Evaluation UI reads their `summary.json`; restart is not needed. PNG charts and per-observation JSONL are also saved. Never upload real raw observations to a public repository.

## Train a learned extractor

```bash
# CRF on your annotated real documents
docker compose run --rm tools python scripts/train_ner.py --manifest /datasets/documents.jsonl --engine crf --output /models/ner-crf

# LiLT + XLM-R from the paper's model family (needs internet, more RAM and time)
docker compose run --rm tools python scripts/train_ner.py --manifest /datasets/documents.jsonl --engine transformer --base-model nielsr/lilt-xlm-roberta-base --output /models/ner-transformer --epochs 15
```

For a text-only transformer comparator use `--base-model FacebookAI/xlm-roberta-base`. This fine-tunes a token-classification head; a general pretrained checkpoint alone is not a KYC extractor. The trainer preserves provenance and selects transformer checkpoints by validation loss. Set `KYC_NER_ENGINE=transformer` and recreate backend to use it. Never change provenance labels to make synthetic data appear real.

## OCR and field extraction

```bash
docker compose run --rm tools python -m app.evaluation.run ocr --manifest /datasets/documents.jsonl --output /evaluations/ocr_test

docker compose run --rm tools python -m app.evaluation.run fields --manifest /datasets/documents.jsonl --engine crf --output /evaluations/crf_test

docker compose run --rm tools python -m app.evaluation.run fields --manifest /datasets/documents.jsonl --engine transformer --output /evaluations/lilt_test
```

CER is total Levenshtein character edits / total reference characters; WER uses word tokens. OCR normalization is NFC plus whitespace collapse, preserving digits, punctuation and case. CER uses Unicode code points, not Devanagari grapheme clusters, and may exceed 1 due to insertions. Both OCR engines use the same input images.

Field metrics count exact matches per field after NFC, digit and whitespace normalization. A wrong non-empty value gives an FP and an FN. Missing values give an FN; extra values give an FP. `fields` measures annotated-token extraction, not OCR+NER end-to-end accuracy. Report the OCR and NER experiments separately; also inspect application outputs to quantify propagated OCR errors in the failure log. The labelled regex comparator is an **offline baseline**, never a runtime fallback.

## Face detectors and recognition

```bash
docker compose run --rm tools python -m app.evaluation.run face --manifest /datasets/face_pairs.csv --output /evaluations/face_test --target-far 0.01

docker compose run --rm tools python -m app.evaluation.run detectors --manifest /datasets/detection.jsonl --output /evaluations/detectors_test
```

The face run evaluates MTCNN and YuNet with both VGGFace2 and CASIA-WebFace variants. Validation ROC selects the highest TPR satisfying the stated empirical FAR target. Test identities are separate. Test ROC/AUC/EER are descriptive, never used to select the operating threshold. Detection failures count as rejected pairs for decision accuracy/FAR/FRR, and are separately reported; ROC/AUC use successful comparisons only. Report that denominator difference. At very low FAR, many impostor trials are necessary; zero observed false accepts is not zero real risk.

Detector JSONL rows: `id,image,boxes,consent_id`; each box is `[x1,y1,x2,y2]`. IoU≥0.5 with one-to-one greedy matching determines TP/FP/FN; also report latency. Detection recall is distinct from face verification accuracy.

After reviewing validation evidence, install the calibration for the model you actually use:

```bash
docker compose run --rm tools python -c "import shutil; shutil.copy('/evaluations/face_test/calibration-mtcnn-vggface2.json','/models/calibration.json')"
docker compose up -d --force-recreate backend
```

This is a deliberate calibration change, not automatic test-set tuning. Runtime checks detector, weight SHA, split, counts and threshold. The continuity threshold and landmark thresholds remain development settings requiring separate validation; the face ROC calibrates the final matching threshold only.

## Liveness

```bash
docker compose run --rm tools python -m app.evaluation.run liveness --manifest /datasets/liveness_trials.jsonl --output /evaluations/liveness_test
```

APCER = accepted attack trials / attack trials (reported separately for each attack type). BPCER = rejected bona fide trials / bona fide trials. Run multiple participants and repeats; retain failures. This academic trial protocol does not certify ISO PAD compliance.

## Failure analysis

Fill `datasets/failure_cases.csv` with at least 20 **observed** cases. Link pseudonymous case/trial IDs, ground truth, component, actual prediction, root cause and proposed fix. Do not present the scenario list in `FAILURE_ANALYSIS.md` as 20 collected observations. The project report deliberately leaves real benchmark numbers pending until these commands have been run on consented evidence.
