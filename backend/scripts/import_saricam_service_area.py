import argparse
import hashlib
import json
from pathlib import Path

from sqlalchemy import text

from backend.app.db.database import engine


SERVICE_AREA_CODE = "SARICAM"
SERVICE_AREA_NAME = "Sarıçam"

SOURCE = (
    "OpenStreetMap relation 1249247 via Nominatim; "
    "retrieved 2026-09-11"
)

GEOJSON_PATH = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "data"
    / "service_areas"
    / "saricam_osm_relation_1249247.geojson"
)


def load_geometry() -> tuple[str, str]:
    raw_bytes = GEOJSON_PATH.read_bytes()

    sha256 = hashlib.sha256(
        raw_bytes
    ).hexdigest()

    data = json.loads(
        raw_bytes.decode("utf-8")
    )

    features = data.get(
        "features",
        [],
    )

    if len(features) != 1:
        raise RuntimeError(
            "Expected exactly one GeoJSON feature."
        )

    feature = features[0]

    properties = feature.get(
        "properties",
        {},
    )

    osm_type = properties.get(
        "osm_type"
    )

    osm_id = properties.get(
        "osm_id"
    )

    if osm_type != "relation":
        raise RuntimeError(
            f"Unexpected OSM type: {osm_type}"
        )

    if osm_id != 1249247:
        raise RuntimeError(
            f"Unexpected OSM relation id: {osm_id}"
        )

    geometry = feature.get(
        "geometry"
    )

    if not geometry:
        raise RuntimeError(
            "GeoJSON geometry is missing."
        )

    geometry_json = json.dumps(
        geometry,
        ensure_ascii=False,
    )

    return geometry_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expected-db",
        required=True,
        help=(
            "Database name that this script "
            "is explicitly allowed to modify."
        ),
    )

    args = parser.parse_args()

    geometry_json, file_sha256 = (
        load_geometry()
    )

    validation_query = text(
        """
        WITH g AS (
            SELECT
                ST_SetSRID(
                    ST_GeomFromGeoJSON(
                        :geometry_json
                    ),
                    4326
                ) AS geom
        )
        SELECT
            ST_IsValid(geom)
                AS is_valid,

            ST_IsValidReason(geom)
                AS valid_reason,

            ST_GeometryType(geom)
                AS geometry_type,

            ST_SRID(geom)
                AS srid

        FROM g
        """
    )

    insert_query = text(
        """
        INSERT INTO service_areas (
            code,
            name,
            source,
            geometry
        )
        VALUES (
            :code,
            :name,
            :source,

            ST_Multi(
                ST_SetSRID(
                    ST_GeomFromGeoJSON(
                        :geometry_json
                    ),
                    4326
                )
            )
        )

        ON CONFLICT (code)
        DO UPDATE SET
            name = EXCLUDED.name,
            source = EXCLUDED.source,
            geometry = EXCLUDED.geometry

        RETURNING id
        """
    )

    inspect_query = text(
        """
        SELECT
            id,
            code,
            name,
            source,

            ST_GeometryType(geometry)
                AS geometry_type,

            ST_SRID(geometry)
                AS srid,

            ST_IsValid(geometry)
                AS is_valid,

            ST_NPoints(geometry)
                AS point_count,

            ST_Area(
                geometry::geography
            ) / 1000000.0
                AS area_km2

        FROM service_areas
        WHERE code = :code
        """
    )

    with engine.begin() as connection:
        database_name = connection.execute(
            text(
                "SELECT current_database()"
            )
        ).scalar_one()

        if (
            database_name
            != args.expected_db
        ):
            raise RuntimeError(
                "SAFETY STOP: Connected to "
                f"'{database_name}' but "
                f"--expected-db was "
                f"'{args.expected_db}'."
            )

        validation = connection.execute(
            validation_query,
            {
                "geometry_json":
                    geometry_json
            },
        ).one()

        if not validation.is_valid:
            raise RuntimeError(
                "Geometry is invalid: "
                f"{validation.valid_reason}"
            )

        if validation.srid != 4326:
            raise RuntimeError(
                "Geometry SRID is not 4326."
            )

        if validation.geometry_type not in (
            "ST_Polygon",
            "ST_MultiPolygon",
        ):
            raise RuntimeError(
                "Unexpected geometry type: "
                f"{validation.geometry_type}"
            )

        inserted_id = connection.execute(
            insert_query,
            {
                "code":
                    SERVICE_AREA_CODE,

                "name":
                    SERVICE_AREA_NAME,

                "source":
                    SOURCE,

                "geometry_json":
                    geometry_json,
            },
        ).scalar_one()

        row = connection.execute(
            inspect_query,
            {
                "code":
                    SERVICE_AREA_CODE
            },
        ).one()

    print(
        "database       =",
        database_name,
    )

    print(
        "service_area_id=",
        inserted_id,
    )

    print(
        "code           =",
        row.code,
    )

    print(
        "name           =",
        row.name,
    )

    print(
        "source         =",
        row.source,
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
        "is_valid       =",
        row.is_valid,
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
        "geojson_sha256 =",
        file_sha256,
    )

    print()
    print(
        "STATUS: "
        "SARICAM_SERVICE_AREA_IMPORTED"
    )


if __name__ == "__main__":
    main()