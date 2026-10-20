"""Download explicit upstream assets, verify loadability, record SHA-256 hashes.
No alternate URL/model is attempted if a requested download fails.
"""

import hashlib, json, os, secrets, shutil, urllib.request
from pathlib import Path
from app.core.config import Settings

ASSETS = {
    "facenet-vggface2.pt": "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt",
    "facenet-casia-webface.pt": "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    settings = Settings()
    root = settings.weights_dir
    root.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    previous = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else json.loads(Path("data/model-assets.json").read_text())
    )
    manifest = {}
    for name, url in ASSETS.items():
        dest = root / name
        if not dest.is_file():
            print(f"Downloading {name}", flush=True)
            temporary = root / (name + ".part")
            try:
                with (
                    urllib.request.urlopen(url, timeout=120) as response,
                    temporary.open("wb") as target,
                ):
                    shutil.copyfileobj(response, target)
                if temporary.stat().st_size < 100_000:
                    raise RuntimeError(
                        "Downloaded file is too small; possible HTML/LFS pointer."
                    )
                temporary.replace(dest)
            finally:
                temporary.unlink(missing_ok=True)
        digest = sha(dest)
        if name in previous and digest != previous[name]["sha256"]:
            raise RuntimeError(
                f"{name}: checksum changed. Inspect/remove this asset before retrying."
            )
        manifest[name] = {"url": url, "sha256": digest, "bytes": dest.stat().st_size}
        # Save each successful transfer so an interrupted setup can resume safely.
        manifest_path.write_text(json.dumps({**previous, **manifest}, indent=2))
    import easyocr

    print("Preparing EasyOCR Nepali + English weights", flush=True)
    easyocr.Reader(
        ["ne", "en"],
        gpu=False,
        model_storage_directory=str(root / "easyocr"),
        download_enabled=True,
        verbose=False,
    )
    for asset in (root / "easyocr").glob("*.pth"):
        manifest[str(asset.relative_to(root))] = {
            "sha256": sha(asset),
            "bytes": asset.stat().st_size,
            "source": "EasyOCR 1.7.2 model registry (upstream MD5 checked by EasyOCR)",
        }
    seed = Path("/opt/seed/ner-crf")
    if not (root / "ner-crf/model.crfsuite").is_file():
        if not seed.is_dir():
            raise RuntimeError(
                "Bundled CRF missing. Run scripts/make_seed.py and scripts/train_ner.py first."
            )
        shutil.copytree(seed, root / "ner-crf", dirs_exist_ok=True)
    key = settings.data_dir / "reviewer-key"
    if not key.exists():
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(36))
    from app.vision.face_verifier import FaceEngine
    import numpy as np

    # Validate both recognition weight sets and both detector adapters before success.
    for detector in ("mtcnn", "yunet"):
        for weights in ("vggface2", "casia-webface"):
            configured = settings.model_copy(
                update={"detector": detector, "face_weights": weights}
            )
            engine = FaceEngine(configured)
            engine.detect(np.zeros((320, 320, 3), np.uint8))
            del engine
    from app.services.registry import Registry

    registry = Registry(settings)
    status = registry.status()
    registry.close()
    if not status["ready"]:
        raise RuntimeError(json.dumps(status["components"]))
    manifest_path.write_text(json.dumps({**previous, **manifest}, indent=2))
    print(
        "All selected models loaded. Both FaceNet variants and detectors checked. No silent substitutions.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"MODEL SETUP FAILED: {error}", flush=True)
        raise SystemExit(1)
