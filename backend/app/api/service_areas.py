import json

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    status,
)
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.schemas.service_area import (
    ServiceAreaFeatureResponse,
    ServiceAreaProperties,
)


router = APIRouter(
    prefix="/service-areas",
    tags=["service-areas"],
)


@router.get(
    "/{service_area_code}",
    response_model=ServiceAreaFeatureResponse,
)
def get_service_area(
    service_area_code: str = Path(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
    db: Session = Depends(get_db),
):
    normalized_code = (
        service_area_code
        .strip()
        .upper()
    )

    row = db.execute(
        text(
            """
            SELECT
                code,
                name,
                source,
                ST_AsGeoJSON(
                    geometry
                ) AS geometry_geojson

            FROM service_areas

            WHERE code = :code
            """
        ),
        {
            "code": normalized_code
        },
    ).mappings().one_or_none()

    if row is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=(
                "Service area not found."
            ),
        )

    geometry = json.loads(
        row["geometry_geojson"]
    )

    return ServiceAreaFeatureResponse(
        properties=(
            ServiceAreaProperties(
                code=row["code"],
                name=row["name"],
                source=row["source"],
            )
        ),
        geometry=geometry,
    )