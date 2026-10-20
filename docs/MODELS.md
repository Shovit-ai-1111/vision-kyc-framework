# Models, weights and honest scope

| Task | Model / algorithm | Provisioning |
|---|---|---|
| Face detection | facenet-pytorch MTCNN; OpenCV YuNet ONNX | MTCNN component weights are packaged by facenet-pytorch; YuNet downloaded by setup |
| Face verification | InceptionResnetV1, VGGFace2 and CASIA-WebFace pretrained variants | Official facenet-pytorch release assets |
| Face landmarks | MediaPipe Face Landmarker task | Official Google model asset |
| OCR | Tesseract LSTM `nep+eng` | Debian language packages in Docker image |
| OCR comparison | EasyOCR CRAFT + Devanagari recognition (`ne,en`) | EasyOCR 1.7.2 registry assets |
| Default fields | CRFsuite linear-chain CRF | Included trained synthetic-seed checkpoint + provenance |
| Transformer fields | LiLT with XLM-R base, or XLM-R token classification | Train using `scripts/train_ner.py`; no fabricated fine-tuned weights included |
| Document type | Bilingual heading rules | Explainable rules; not represented as a trained classifier |
| Presentation checks | Active blink/head movement and face continuity | Not a trained anti-spoofing/PAD classifier |

Face preprocessing is identical during runtime and evaluation: detected box + 10% margin, resize to 160 × 160, BGR→RGB, `(pixel−127.5)/128`, 512-dimensional L2-normalized embedding. No eye rotation or recognition-engine substitution occurs silently. Cosine similarity is a score, not an identity probability.

Both detector adapters and both recognition variants were loaded and exercised locally (see `model-smoke-results.json`). facenet-pytorch 2.6.0's package metadata pins older Torch versions; the Dockerfile deliberately installs that package without dependencies after Torch 2.6.0 CPU / torchvision 0.21.0 and the explicit requirements. Real inference smoke checks passed with that Torch pair. The dependency-metadata mismatch is documented, not hidden. Two OpenCV distributions are required by upstream OCR/MediaPipe packages; both are pinned to 4.11.0.86 to avoid conflicting versions.

`backend/data/model-assets.json` records SHA-256 hashes from the fetched release assets. Setup compares downloads with that manifest and writes the local manifest. EasyOCR additionally checks its upstream registry MD5. Large weights are excluded from the ZIP; setup downloads and checks them. Runtime OCR/model adapters never download or select another model when an asset is missing.

## Relation to the cited paper

Neupane et al. (CHiPSAL 2025) investigate OCR-based information extraction from Nepali notices using LiLT and language models. Their dataset and fields are not citizenship/PAN identity verification. This project adopts token/layout-based extraction and provides a LiLT fine-tuning path; it does not claim to reproduce their results or reuse their reported scores as project measurements.

Sources:
- https://aclanthology.org/2025.chipsal-1.13/
- https://github.com/timesler/facenet-pytorch
- https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
- https://github.com/JaidedAI/EasyOCR
- https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
- https://huggingface.co/nielsr/lilt-xlm-roberta-base
- https://huggingface.co/docs/transformers/model_doc/lilt
