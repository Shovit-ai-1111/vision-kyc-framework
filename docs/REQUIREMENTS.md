# Professor feedback: implementation and evidence status

| Requested change | Delivered | Still required from the team |
|---|---|---|
| Structured analysis / referenced paper | OCR tokens and boxes, learned BIO extraction, LiLT training adapter, typed field validation | Annotate real Nepali IDs and fine-tune/evaluate LiLT |
| Buggy face comparison | Actual FaceNet inference; no grayscale descriptor; one-face policy | Evaluate error rates on consented ID/selfie pairs |
| Visible MediaPipe/liveness | 478 landmark overlay, EAR/head proxy charts, per-step events, continuity check | Real camera validation and print/screen/replay trials |
| Document crop examples | Live before/after previews, actual synthetic-layout crop outputs and diagrams | Add anonymized real ID examples with permission |
| More than one detector/model | MTCNN vs YuNet; FaceNet VGGFace2 vs CASIA-WebFace | Run comparative study; 3-member lighting pilot |
| Remove fallbacks | Missing models return 503; no alternative engine or fake vector is selected | Keep weights installed and inspect readiness |
| Charts | Dynamic UI and evaluation-generated PNG/JSON charts | Generate real-data plots using included tools |
| 10–50 documents | Dataset schema, consent/split checks and annotation helper | Collect actual consented documents; seed text does not meet this requirement |
| Two bilingual OCR engines / CER | Tesseract and EasyOCR adapters; micro CER/WER evaluator | Correct gold transcriptions and run both engines |
| NLP extraction / per-field metrics | Trained CRF seed model; transformer adapter; per-field P/R/F1 | Real-data labels and held-out results |
| Video flow | Server-observed live frames and challenge-bound final face | Validate with target laptop/webcam; no attended agent video call included |
| Consistency | Front/back/entered/secondary comparisons and reasons | Add real examples and adjudicate cross-script/date ambiguity |
| ≥50 people / ROC threshold | Group-disjoint pair validation; calibration artifacts bound to model checksum | Recruit participants, generate genuine/impostor pairs, select validation threshold |
| Attack evaluation | Trial export and APCER/BPCER calculation | Collect labelled print, phone photo and replay trials |
| Reviewer interface | Approve/reject/correct with access key, revision checking and audit history | Assign actual reviewers and protect deployment |
| 20 failure cases | Failure-log template and scenario analysis | Fill at least 20 observed dataset failure cases with evidence |
| Meaningful Git history | Contribution guidance and assistance disclosure | Each member must commit actual work over time; history cannot be reconstructed honestly from a ZIP |

This package is an implemented research pipeline with explicit gaps, not a claim that the entire empirical resubmission is complete.
