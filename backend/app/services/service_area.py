from sqlalchemy import text
from sqlalchemy.orm import Session


SERVICE_AREA_CODE = "SARICAM"


class ServiceAreaNotConfiguredError(RuntimeError):
    pass


def is_location_within_service_area(
    db: Session,
    *,
    latitude: float,
    longitude: float,
    service_area_code: str = SERVICE_AREA_CODE,
) -> bool:
    result = db.execute(
        text(
            """
            SELECT ST_Covers(
                geometry,
                ST_SetSRID(
                    ST_MakePoint(
                        :longitude,
                        :latitude
                    ),
                    4326
                )
            )
            FROM service_areas
            WHERE code = :service_area_code
            """
        ),
        {
            "longitude": longitude,
            "latitude": latitude,
            "service_area_code": service_area_code,
        },
    ).scalar_one_or_none()

    if result is None:
        raise ServiceAreaNotConfiguredError(
            f"Service area '{service_area_code}' "
            "is not configured."
        )

    return bool(result)