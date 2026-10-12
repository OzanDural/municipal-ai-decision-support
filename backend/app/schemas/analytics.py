from pydantic import BaseModel
from datetime import (
    date,
    datetime,
)
from pydantic import BaseModel

class AnalyticsSummaryResponse(BaseModel):
    total_complaints: int

    auto_accepted: int
    review_pending: int
    reviewed: int

    human_review_total: int
    corrected_by_human: int

    auto_acceptance_rate: float
    human_review_rate: float
    correction_rate: float


class CategoryDistributionItem(BaseModel):
    category: str
    count: int
    percentage: float
class TrendItem(BaseModel):
    date: date
    count: int

class NearbyComplaintItem(BaseModel):
    id: int
    application_no: str
    complaint_text: str

    category: str
    status: str

    latitude: float
    longitude: float

    distance_meters: float


class NearbyComplaintsResponse(BaseModel):
    center_latitude: float
    center_longitude: float

    radius_meters: float
    category_filter: str | None

    count: int

    results: list[NearbyComplaintItem]
class MapPointItem(BaseModel):
    id: int
    application_no: str

    complaint_text: str
    neighborhood: str | None

    category: str
    status: str

    latitude: float
    longitude: float

    created_at: datetime
class NeighborhoodAnalyticsItem(BaseModel):
    neighborhood: str

    count: int
    percentage: float

    dominant_category: str
    dominant_category_count: int

    category_counts: dict[str, int]
class ModelHumanCategoryItem(BaseModel):
    category: str

    reviewed: int
    confirmed: int
    corrected: int

    correction_rate: float


class CorrectionFlowItem(BaseModel):
    predicted_category: str
    final_category: str
    count: int


class ModelHumanComparisonResponse(BaseModel):
    reviewed: int

    confirmed: int
    corrected: int

    agreement_rate: float
    correction_rate: float

    by_predicted_category: list[
        ModelHumanCategoryItem
    ]

    correction_flows: list[
        CorrectionFlowItem
    ]