from datetime import date
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from backend.app.models.enums import (
    ComplaintCategory,
)


ManagerAssistantAnalysisType = Literal[
    "summary",
    "category_distribution",
    "trend",
    "neighborhoods",
    "model_human",
    "executive_insight",
    "nearby",
]


class ManagerAssistantRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    analysis_type: (
        ManagerAssistantAnalysisType | None
    ) = None

    from_date: date | None = None
    to_date: date | None = None

    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
    )

    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
    )

    radius_meters: float = Field(
        default=1000,
        ge=1,
        le=10000,
    )

    category: ComplaintCategory | None = None

    @field_validator("message")
    @classmethod
    def normalize_message(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Message cannot be empty."
            )

        return normalized

    @model_validator(mode="after")
    def validate_context(
        self,
    ):
        if (
            self.from_date
            and self.to_date
            and self.from_date
            > self.to_date
        ):
            raise ValueError(
                "from_date cannot be after to_date."
            )

        if (
            self.analysis_type == "nearby"
            and (
                self.latitude is None
                or self.longitude is None
            )
        ):
            raise ValueError(
                (
                    "Nearby analysis requires "
                    "latitude and longitude."
                )
            )

        return self


class ManagerAssistantResponse(BaseModel):
    answer: str

    source: Literal[
        "gemini",
        "deterministic_fallback",
    ] = "gemini"

    notice: str | None = None
