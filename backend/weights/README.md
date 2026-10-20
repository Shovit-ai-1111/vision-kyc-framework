# Explicit model assets

The ZIP includes `ner-crf/model.crfsuite` and its provenance. This is a trained synthetic-seed model, not a missing/empty placeholder.

Large FaceNet, YuNet, MediaPipe and EasyOCR assets are provisioned by `scripts/setup_models.py` into the Docker `models` volume. Expected upstream asset hashes are recorded in `data/model-assets.json`. There is no runtime download or model substitution.

Fine-tuned transformer weights and held-out calibration are intentionally absent until your real-data experiments produce them. Setting the transformer engine without a checkpoint gives `model_not_loaded`; missing calibration is visible and prevents an automatic pass.
