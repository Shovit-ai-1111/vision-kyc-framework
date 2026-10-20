# Dataset preparation

## Document study: 10–50 consented images

Collect citizenship front/back, passport portrait pages and PAN images. Record document family, language, lighting, capture device and a pseudonymous person ID. Store consent forms separately. Do not put real IDs, consent forms or biometric data in public Git repositories.

Split by **person**, not by image: all front/back images and augmentations for one person must remain in the same split. A small starting plan is 60% train / 20% validation / 20% test, stratified by document family where feasible. With only 10–50 images the estimates are preliminary; report denominators and uncertainty.

Each JSONL row includes `id`, `subject_id`, `consent_id`, `data_kind=consented_real`, `split`, `document_type`, `side`, `image` (relative to manifest), `width`, `height`, `text` (corrected full transcription), `tokens` (text, pixel box, line, confidence), `labels` (one BIO label per token), and `fields` (correct values). Keep gold transcription separate from raw OCR; correct annotation errors manually. Annotators should independently label a subset, resolve disagreements and record changes.

```bash
docker compose run --rm tools python scripts/annotate_document.py --image /datasets/images/p001-front.jpg --id doc001 --subject p001 --consent-id consent001 --split train --type citizenship --side front --output /datasets/doc001.jsonl
```

Correct the draft transcription, labels and fields, then concatenate approved JSONL rows into `/datasets/documents.jsonl`. `backend/data/seed.jsonl` demonstrates the full schema but contains **synthetic text, not real document images**. Token labels: O and B-/I- for NAME, FATHER_NAME, DOB, DOCUMENT_NUMBER, DISTRICT, ISSUE_DATE, SURNAME, GIVEN_NAMES. The last two are combined for passports. Use actual bounding boxes; EasyOCR word boxes are approximated from line boxes and marked accordingly.

## Face study: at least 50 people

Use consented ID-portrait/selfie pairs. Face-pair images must be **portrait crops**, because the evaluation engine expects one face per image. Keep validation and test identities disjoint; for example, 25 validation and 25 test participants, each with multiple captures. Pretrained FaceNet is frozen; this study is threshold selection/evaluation, not FaceNet training.

CSV fields: `pair_id,subject_a,subject_b,image_a,image_b,split,lighting,consent_id`. Genuine status is derived from equal subject IDs; impostor pairs use different people. Never compare an image with itself. Generate genuine and impostor pairs separately within each split, not across splits. Do not drop detector failures. The CLI rejects identity and image leakage. Calibration requires at least 20 validation identities, 50 study identities, and 25 genuine + 25 impostor successful validation comparisons. These minimums are guards, not a claim of statistically precise low-FAR estimation.

For the three-member lighting pilot, record each team member in normal indoor light, dim light and backlight, ideally five captures per condition. Pair each selfie with that person's ID portrait and construct cross-member impostor pairs. Use `--allow-small` to label this a **small pilot**; it cannot produce an acceptable full-study calibration.

## Liveness trials

For each participant, run randomized online challenges under bona fide, printed-photo, phone-photo and recorded-phone-video conditions. Record failed/abandoned attempts too. A moved print can exhibit head motion; an adaptive replay can defeat simple active checks. These are expected limitations to investigate, not assume solved.

In Reviewer → Liveness events, download the trial JSON. It contains the server result and event trace without document previews. Create a JSONL row with `id,subject_id,consent_id,attack_type,record`. `record` is relative to the manifest. Allowed attack types: `bona_fide`, `print`, `phone_photo`, `phone_replay`. Wait for timeout cleanup before exporting an abandoned session as FAILED. Keep a separate operator log/video evidence with permission.

The ZIP's sample images are marked INVALID and use a public-domain NASA/skimage test portrait on fictional text layouts. They are extraction examples only. They must not be included in real-data performance claims.
