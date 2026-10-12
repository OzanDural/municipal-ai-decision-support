from datetime import datetime

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from backend.app.data.neighborhoods import (
    SARICAM_NEIGHBORHOOD_SET,
)
from backend.app.models.enums import (
    ComplaintCategory,
)


class ComplaintCreate(BaseModel):
    complaint_text: str = Field(
        min_length=5,
        max_length=3000,
    )

    neighborhood: str = Field(
        min_length=2,
        max_length=100,
    )

    street_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
    )

    address_text: str | None = Field(
        default=None,
        max_length=500,
    )

    latitude: float = Field(
        ge=-90,
        le=90,
    )

    longitude: float = Field(
        ge=-180,
        le=180,
    )

    @field_validator(
        "complaint_text"
    )
    @classmethod
    def validate_complaint_text(
        cls,
        value: str,
    ) -> str:
        cleaned = " ".join(
            value.split()
        )

        if len(cleaned) < 5:
            raise ValueError(
                "Complaint text is too short."
            )

        return cleaned

    @field_validator(
        "neighborhood",
        mode="before",
    )
    @classmethod
    def validate_neighborhood(
        cls,
        value: str | None,
    ) -> str:
        if value is None:
            raise ValueError(
                "Neighborhood is required."
            )

        cleaned = " ".join(
            value.split()
        )

        if not cleaned:
            raise ValueError(
                "Neighborhood is required."
            )

        if (
            cleaned
            not in SARICAM_NEIGHBORHOOD_SET
        ):
            raise ValueError(
                "Neighborhood is not a valid "
                "Sarıçam neighborhood."
            )

        return cleaned

    @field_validator(
        "street_name",
        mode="before",
    )
    @classmethod
    def normalize_street_name(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned = " ".join(
            value.split()
        )

        if not cleaned:
            return None

        return cleaned

    @field_validator(
        "address_text",
        mode="before",
    )
    @classmethod
    def normalize_address_text(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned = " ".join(
            value.split()
        )

        if not cleaned:
            return None

        return cleaned


class ComplaintCreateResponse(BaseModel):
    id: int
    application_no: str

    neighborhood: str | None
    street_name: str | None
    address_text: str | None

    predicted_category: str
    secondary_category: str

    model_score: float
    second_model_score: float
    score_margin: float

    review_required: bool
    review_reason: str | None

    status: str
    model_version: str

    latitude: float
    longitude: float

    created_at: datetime


class ComplaintListItem(BaseModel):
    id: int
    application_no: str
    complaint_text: str

    neighborhood: str | None
    street_name: str | None
    address_text: str | None

    predicted_category: str
    secondary_category: str

    model_score: float
    score_margin: float

    review_required: bool
    review_reason: str | None

    final_category: str | None
    was_corrected: bool | None

    status: str

    latitude: float
    longitude: float

    created_at: datetime

class ComplaintPageResponse(BaseModel):
    items: list[ComplaintListItem]

    total: int

    page: int
    page_size: int
    total_pages: int

class ComplaintDetailResponse(BaseModel):
    id: int
    application_no: str
    complaint_text: str

    neighborhood: str | None
    street_name: str | None
    address_text: str | None

    predicted_category: str
    secondary_category: str

    model_score: float
    second_model_score: float
    score_margin: float

    model_version: str

    review_required: bool
    review_reason: str | None

    final_category: str | None
    was_corrected: bool | None

    reviewed_by: int | None
    reviewed_at: datetime | None

    status: str

    latitude: float
    longitude: float

    created_at: datetime
    updated_at: datetime


class ComplaintReviewRequest(BaseModel):
    final_category: ComplaintCategory


class ComplaintReviewResponse(BaseModel):
    id: int
    application_no: str

    predicted_category: str
    final_category: str

    was_corrected: bool

    reviewed_by: int
    reviewed_at: datetime

    review_required: bool
    status: str