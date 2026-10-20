# Failure analysis: observed limits and collection worksheet

Observed during implementation: OCR can read texture inside a portrait as text; the document pipeline now masks detected portrait boxes before OCR. Synthetic CRF training initially learned day-number shortcuts for date fields; features now normalize digits to shapes. These findings motivated fixes, but synthetic tests do not establish generalization to real Nepali documents. Record any remaining extraction confusion in the real dataset study.

The following are **analysis scenarios, not 20 empirical dataset observations**. Use them to collect the minimum 20 real failure cases in `datasets/failure_cases.csv`.

| # | Scenario | Mechanism / expected handling |
|---|---|---|
| 1 | Missing FaceNet weights | Explicit model-not-loaded; no pixel descriptor |
| 2 | Missing Nepali OCR pack | Bilingual OCR readiness fails |
| 3 | Missing trained transformer | Selected NER engine fails; no CRF substitution |
| 4 | Blur | OCR confidence/field completeness declines; investigate threshold |
| 5 | Glare | Lost characters and portrait features; recapture |
| 6 | Rotated document | No silent orientation search; supply upright image |
| 7 | Cropped document edges | Photo rectification fails; deliberate scan mode only for actual flat scans |
| 8 | Devanagari vowel/diacritic error | CER and exact-field F1 reveal OCR propagation |
| 9 | Handwritten fields | Generic OCR may fail; annotate separately |
| 10 | Unseen ID template | Heading/CRF context may fail; real layout training needed |
| 11 | Front/back belong to different people | Overlapping field mismatches trigger review/rejection |
| 12 | Different scripts for the same name | Mark uncertain; no guessed transliteration equality |
| 13 | BS vs AD dates | Mark uncertain; no guessed calendar conversion |
| 14 | Invalid date strings | Inspect extracted field formats and annotation |
| 15 | Multiple portraits/ghost portrait | No largest-face assumption; explicit count issue |
| 16 | Tiny ID portrait | Low-resolution review; quantify face false rejection |
| 17 | Backlit selfie | Quality/embedding failures; evaluate by lighting |
| 18 | Eyeglasses or partly closed eyes | EAR calibration may fail; collect BPCER cases |
| 19 | Missed fast blink on slow CPU | Deliberate slow close/open prompt; log round-trip time |
| 20 | Moved printed photograph | May mimic head movement; blink helps but is not certified PAD |
| 21 | Phone showing a face photo | Evaluate acceptance; screen artifacts are not explicitly classified |
| 22 | Adaptive prerecorded video | Can defeat simple challenge systems; record APCER honestly |
| 23 | Byte-identical repeated frame | Rejected as replay; re-encoded replay may evade hash checks |
| 24 | Face substitution during challenge | Continuity threshold can detect some swaps; threshold still needs calibration |
| 25 | Convincing forged PAN with no sample marker | Structural consistency cannot prove issuance; issuer verification is needed |
| 26 | Interrupted/restarted session | In-memory challenge invalidates; restart with evidence retained |

Do not tune thresholds on the final test set after inspecting failures. Keep a new validation cycle and reserve a fresh test set if changes are substantial.
