import json
from pathlib import Path

from sqlalchemy import text

from backend.app.db.database import engine


GEOJSON_PATH = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "data"
    / "service_areas"
    / "saricam_osm_relation_1249247.geojson"
)


def main() -> None:
    with GEOJSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        feature_collection = json.load(file)

    features = feature_collection.get(
        "features",
        [],
    )

    if len(features) != 1:
        raise RuntimeError(
            "Expected exactly one GeoJSON feature."
        )

    feature = features[0]
    geometry = feature.get("geometry")

    if not geometry:
        raise RuntimeError(
            "GeoJSON feature has no geometry."
        )

    geometry_json = json.dumps(
        geometry,
        ensure_ascii=False,
    )

    query = text(
        """
        WITH source_geometry AS (
            SELECT
                ST_SetSRID(
                    ST_GeomFromGeoJSON(:geometry_json),
                    4326
                ) AS geom
        ),
        normalized AS (
            SELECT
                ST_Multi(geom) AS geom
            FROM source_geometry
        )
        SELECT
            ST_IsValid(geom) AS is_valid,

            ST_IsValidReason(geom)
                AS valid_reason,

            ST_GeometryType(geom)
                AS geometry_type,

            ST_SRID(geom)
                AS srid,

            ST_NPoints(geom)
                AS point_count,

            ST_Area(
                geom::geography
            ) / 1000000.0
                AS area_km2,

            ST_XMin(
                ST_Extent(geom)
            ) AS min_lon,

            ST_XMax(
                ST_Extent(geom)
            ) AS max_lon,

            ST_YMin(
                ST_Extent(geom)
            ) AS min_lat,

            ST_YMax(
                ST_Extent(geom)
            ) AS max_lat,

            ST_X(
                ST_PointOnSurface(geom)
            ) AS inside_lon,

            ST_Y(
                ST_PointOnSurface(geom)
            ) AS inside_lat

        FROM normalized
        GROUP BY geom
        """
    )

    with engine.connect() as connection:
        database_name = connection.execute(
            text(
                "SELECT current_database()"
            )
        ).scalar_one()

        if database_name != "belediye_ai_test":
            raise RuntimeError(
                "SAFETY STOP: Expected "
                "belediye_ai_test but connected to "
                f"{database_name}."
            )

        row = connection.execute(
            query,
            {
                "geometry_json":
                    geometry_json
            },
        ).one()

    print(
        "database       =",
        database_name,
    )

    print(
        "is_valid       =",
        row.is_valid,
    )

    print(
        "valid_reason   =",
        row.valid_reason,
    )

    print(
        "geometry_type  =",
        row.geometry_type,
    )

    print(
        "srid           =",
        row.srid,
    )

    print(
        "point_count    =",
        row.point_count,
    )

    print(
        "area_km2       =",
        float(row.area_km2),
    )

    print(
        "min_lon        =",
        float(row.min_lon),
    )

    print(
        "max_lon        =",
        float(row.max_lon),
    )

    print(
        "min_lat        =",
        float(row.min_lat),
    )

    print(
        "max_lat        =",
        float(row.max_lat),
    )

    print(
        "inside_lon     =",
        float(row.inside_lon),
    )

    print(
        "inside_lat     =",
        float(row.inside_lat),
    )

    if not row.is_valid:
        raise RuntimeError(
            "Service-area geometry is invalid: "
            f"{row.valid_reason}"
        )

    if (
        row.geometry_type
        != "ST_MultiPolygon"
    ):
        raise RuntimeError(
            "Normalized geometry must be "
            "ST_MultiPolygon."
        )

    if row.srid != 4326:
        raise RuntimeError(
            "Unexpected SRID."
        )

    print()
    print(
        "STATUS: "
        "SARICAM_GEOJSON_VALID"
    )


if __name__ == "__main__":
    main()