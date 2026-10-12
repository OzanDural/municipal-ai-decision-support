from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    complaint_text: str = Field(
        min_length=5,
        max_length=3000,
    )

    @field_validator("complaint_text")
    @classmethod
    def validate_complaint_text(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if len(cleaned) < 5:
            raise ValueError(
                "Complaint text is too short."
            )

        return cleaned


class PredictionResponse(BaseModel):
    predicted_category: str
    secondary_category: str

    model_score: float
    second_model_score: float
    score_margin: float

    review_required: bool
    review_reason: str | None

    model_version: str