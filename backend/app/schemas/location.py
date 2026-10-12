from pydantic import BaseModel


class NeighborhoodListResponse(BaseModel):
    district_code: str
    district_name: str
    count: int
    neighborhoods: list[str]


class StreetListResponse(BaseModel):
    district_code: str
    district_name: str

    source: str
    retrieved_at: str

    count: int
    streets: list[str]
class NeighborhoodCenter(BaseModel):
    name: str
    latitude: float
    longitude: float


class NeighborhoodCenterListResponse(
    BaseModel
):
    district_code: str
    district_name: str
    count: int
    centers: list[
        NeighborhoodCenter
    ]