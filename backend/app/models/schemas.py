from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

DocumentType = Literal[
    "citizenship", "passport", "pan", "national_id", "driving_license"
]
FIELDS = ("name", "father_name", "dob", "document_number", "district", "issue_date")


class Claims(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=2, max_length=160)
    father_name: str = Field(default="", max_length=160)
    dob: str = Field(default="", max_length=40)
    document_number: str = Field(min_length=3, max_length=60)
    district: str = Field(default="", max_length=100)
    issue_date: str = Field(default="", max_length=40)


class CaseInput(BaseModel):
    document_type: DocumentType
    claims: Claims
    consent: Literal[True]
    cloud_ocr_consent: bool = False


class ApplicantCorrectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    expected_revision: int = Field(ge=0)
    corrections: dict[str, str] = Field(min_length=1, max_length=14)
    reason: str = Field(min_length=5, max_length=500)


class FrameInput(BaseModel):
    sequence: int = Field(ge=0)
    image: str = Field(max_length=12_000_000)


class ReviewInput(BaseModel):
    action: Literal["APPROVE", "REJECT", "CORRECT"]
    reviewer: str = Field(min_length=2, max_length=100)
    reason: str = Field(min_length=10, max_length=1000)
    corrections: dict[str, str] = Field(default_factory=dict)
    expected_revision: int = Field(ge=0)
