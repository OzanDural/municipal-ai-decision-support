from fastapi import APIRouter

from backend.app.data.neighborhoods import (
    SARICAM_NEIGHBORHOODS,
)
from backend.app.data.streets import (
    SARICAM_STREET_DATA,
    SARICAM_STREETS,
)
from backend.app.schemas.location import (
    NeighborhoodListResponse,
    StreetListResponse,
    NeighborhoodCenterListResponse,
)
from backend.app.data.neighborhood_centers import (
    SARICAM_NEIGHBORHOOD_CENTERS,
)

router = APIRouter(
    prefix="/locations",
    tags=["locations"],
)


@router.get(
    "/neighborhoods",
    response_model=NeighborhoodListResponse,
)
def list_neighborhoods():
    neighborhoods = list(
        SARICAM_NEIGHBORHOODS
    )

    return NeighborhoodListResponse(
        district_code="SARICAM",
        district_name="Sarıçam",
        count=len(neighborhoods),
        neighborhoods=neighborhoods,
    )


@router.get(
    "/streets",
    response_model=StreetListResponse,
)
def list_streets():
    streets = list(
        SARICAM_STREETS
    )

    return StreetListResponse(
        district_code="SARICAM",
        district_name="Sarıçam",
        source=str(
            SARICAM_STREET_DATA[
                "source"
            ]
        ),
        retrieved_at=str(
            SARICAM_STREET_DATA[
                "retrieved_at"
            ]
        ),
        count=len(streets),
        streets=streets,
    )
@router.get(
    "/neighborhood-centers",
    response_model=(
        NeighborhoodCenterListResponse
    ),
)
def get_neighborhood_centers():
    return {
        "district_code":
            "SARICAM",

        "district_name":
            "Sarıçam",

        "count":
            len(
                SARICAM_NEIGHBORHOOD_CENTERS
            ),

        "centers":
            list(
                SARICAM_NEIGHBORHOOD_CENTERS
            ),
    }