# API v2

This revision replaces the v1 demo API. React and backend in this ZIP must be deployed together.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health/live` | Process health (Docker healthcheck) |
| GET | `/health/ready` | Selected model readiness; 503 if any is missing |
| GET | `/api/models` | Readiness details, provenance and calibration state |
| POST | `/api/cases` | Create with document_type, claims and consent=true; returns session token |
| GET / DELETE | `/api/cases/{id}` | View/delete own case |
| POST | `/api/cases/{id}/documents` | Multipart image, document_type, front/back side, primary/secondary evidence, flat_scan/photograph input_mode |
| DELETE | `/api/cases/{id}/documents/{document_id}` | Remove evidence before completion; invalidates challenge |
| POST | `/api/cases/{id}/challenge` | Randomized live challenge |
| POST | `/api/cases/{id}/challenge/{challenge_id}/frames` | JSON sequence and base64 image; returns landmarks and step state |
| POST | `/api/cases/{id}/finish` | Compare the server-held last accepted live frame; no caller-supplied final selfie |
| GET | `/api/review/cases` | Reviewer list |
| GET / DELETE | `/api/review/cases/{id}` | Reviewer detail / delete |
| POST | `/api/review/cases/{id}` | APPROVE, REJECT or CORRECT, reviewer name, reason, corrections, expected_revision |
| GET | `/api/evaluations` | Saved aggregate metrics and chart data |

Case routes require `Authorization: Bearer <session_token>`. Review routes require `X-Reviewer-Key`. The key must have at least 24 characters; setup generates one. Session tokens are stored only as hashes. All inference outputs omit retained embeddings. Revision conflicts return 409. Model-load failures return 503 with `error.code=model_not_loaded`. The old demo-mode query/configuration no longer exists.

The live HTTP upload flow cannot cryptographically attest that the client sent a physical webcam frame. A modified client can inject images/video. Randomized temporal tasks raise the difficulty of simple replay but do not solve this threat; include it in limitations.
