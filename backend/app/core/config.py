from pathlib import Path
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_prefix="KYC_", extra="ignore"
    )
    weights_dir: Path = Path("weights")
    data_dir: Path = Path("var")
    evaluations_dir: Path = Path("../evaluations")
    reviewer_key: str = ""
    ocr_engine: Literal["tesseract", "easyocr"] = "tesseract"
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="GEMINI_API_KEY", repr=False
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite", pattern=r"^gemini-[a-zA-Z0-9.\-]+$"
    )
    gemini_enabled: bool = True
    gemini_timeout_seconds: int = Field(20, ge=3, le=60)
    gemini_attempts: int = Field(3, ge=1, le=4)
    ocr_variants: int = Field(5, ge=2, le=5)
    ocr_timeout_seconds: int = Field(12, ge=3, le=30)
    ocr_secondary: bool = False
    ner_engine: Literal["crf", "transformer"] = "crf"
    detector: Literal["mtcnn", "yunet"] = "mtcnn"
    face_weights: Literal["vggface2", "casia-webface"] = "vggface2"
    device: str = "cpu"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    session_ttl_seconds: int = Field(1800, ge=300, le=7200)
    retention_hours: int = Field(24, ge=1, le=168)
    challenge_ttl_seconds: int = Field(120, ge=30, le=240)
    max_image_bytes: int = 8 * 1024 * 1024
    max_request_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 20_000_000
    max_cases: int = 200
    max_active_challenges: int = 8
    # Development settings, never presented as calibrated thresholds.
    similarity_threshold: float = Field(0.70, gt=-1, le=1)
    continuity_threshold: float = Field(0.55, gt=-1, le=1)
    min_field_confidence: float = Field(0.70, ge=0, le=1)
    min_ocr_confidence: float = Field(0.50, ge=0, le=1)
    log_level: str = "INFO"
