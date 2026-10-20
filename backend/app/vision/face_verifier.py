from threading import RLock
import cv2
import numpy as np
from app.core.errors import KycError


class FaceEngine:
    """Explicit detector choice. Both choices use a trained FaceNet embedder."""

    def __init__(self, settings):
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN

        self.torch, self.settings = torch, settings
        self.lock = RLock()
        torch.set_num_threads(2)
        checkpoint = settings.weights_dir / f"facenet-{settings.face_weights}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Model not loaded: {checkpoint.name}. Run model-setup."
            )
        # weights_only prevents arbitrary pickle object loading in this path.
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = InceptionResnetV1(
            classify=True,
            num_classes=8631 if settings.face_weights == "vggface2" else 10575,
        )
        self.model.load_state_dict(weights, strict=True)
        self.model.classify = False
        self.model.eval().to(settings.device)
        if settings.detector == "mtcnn":
            self.detector = MTCNN(
                keep_all=True,
                device=settings.device,
                min_face_size=35,
                thresholds=[0.6, 0.7, 0.85],
            )
        else:
            path = settings.weights_dir / "yunet.onnx"
            if not path.is_file():
                raise FileNotFoundError(
                    "Model not loaded: yunet.onnx. Run model-setup."
                )
            self.detector = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), 0.85, 0.3, 5000
            )
        self.embed(np.full((160, 160, 3), 128, dtype=np.uint8))

    def detect(self, bgr):
        with self.lock:
            if self.settings.detector == "mtcnn":
                boxes, scores, landmarks = self.detector.detect(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), landmarks=True
                )
                if boxes is None:
                    return []
                results = [
                    (b, s, l) for b, s, l in zip(boxes, scores, landmarks) if s >= 0.90
                ]
            else:
                self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
                _, faces = self.detector.detect(bgr)
                results = (
                    []
                    if faces is None
                    else [
                        (
                            np.array([f[0], f[1], f[0] + f[2], f[1] + f[3]]),
                            f[14],
                            f[4:14].reshape(5, 2),
                        )
                        for f in faces
                    ]
                )
        output = []
        for box, score, landmarks in results:
            x1, y1, x2, y2 = box
            box = [
                max(0, int(x1)),
                max(0, int(y1)),
                min(bgr.shape[1], int(x2)),
                min(bgr.shape[0], int(y2)),
            ]
            if min(box[2] - box[0], box[3] - box[1]) >= 30:
                output.append(
                    {
                        "box": box,
                        "confidence": float(score),
                        "landmarks": np.asarray(landmarks).tolist(),
                    }
                )
        return output

    def crop(self, bgr, detection):
        x1, y1, x2, y2 = detection["box"]
        # Fixed crop protocol used by online inference and evaluation.
        mx, my = int((x2 - x1) * 0.10), int((y2 - y1) * 0.10)
        return bgr[
            max(0, y1 - my) : min(bgr.shape[0], y2 + my),
            max(0, x1 - mx) : min(bgr.shape[1], x2 + mx),
        ].copy()

    def one(self, bgr):
        found = self.detect(bgr)
        if len(found) != 1:
            raise KycError(
                "face_count",
                f"Expected one clear face; detected {len(found)}. Remove other faces and improve framing.",
            )
        return self.crop(bgr, found[0]), found[0]

    def embed(self, bgr):
        rgb = cv2.cvtColor(cv2.resize(bgr, (160, 160)), cv2.COLOR_BGR2RGB)
        tensor = (
            self.torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            .float()
            .unsqueeze(0)
        )
        tensor = ((tensor - 127.5) / 128.0).to(self.settings.device)
        with self.lock, self.torch.inference_mode():
            vector = self.model(tensor).cpu().numpy()[0]
        if (
            vector.shape != (512,)
            or not np.isfinite(vector).all()
            or np.linalg.norm(vector) < 1e-8
        ):
            raise KycError(
                "embedding_error", "FaceNet produced an invalid embedding.", 503
            )
        return vector / np.linalg.norm(vector)

    @staticmethod
    def similarity(a, b):
        return float(
            np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
        )
