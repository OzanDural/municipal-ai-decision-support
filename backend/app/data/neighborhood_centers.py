import json
from pathlib import Path

from backend.app.data.neighborhoods import (
    SARICAM_NEIGHBORHOODS,
    SARICAM_NEIGHBORHOOD_SET,
)


DATA_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "data"
    / "locations"
    / "saricam_neighborhood_centers.json"
)


def load_neighborhood_centers():
    if not DATA_PATH.exists():
        raise RuntimeError(
            "Neighborhood center data file "
            "was not found."
        )

    data = json.loads(
        DATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    centers = data.get(
        "centers"
    )

    if not isinstance(
        centers,
        list,
    ):
        raise RuntimeError(
            "Neighborhood center data "
            "is invalid."
        )

    if data.get(
        "unresolved_count"
    ) != 0:
        raise RuntimeError(
            "Neighborhood center data "
            "contains unresolved records."
        )

    if data.get(
        "review_required_count"
    ) != 0:
        raise RuntimeError(
            "Neighborhood center data "
            "contains records requiring review."
        )

    if data.get(
        "count"
    ) != len(centers):
        raise RuntimeError(
            "Neighborhood center count "
            "does not match the data."
        )

    centers_by_name = {}

    for item in centers:
        name = item.get(
            "name"
        )

        latitude = item.get(
            "latitude"
        )

        longitude = item.get(
            "longitude"
        )

        if (
            not isinstance(name, str)
            or name
            not in SARICAM_NEIGHBORHOOD_SET
        ):
            raise RuntimeError(
                "Neighborhood center contains "
                "an invalid neighborhood name."
            )

        if name in centers_by_name:
            raise RuntimeError(
                "Neighborhood center data "
                "contains duplicate names."
            )

        try:
            latitude = float(
                latitude
            )

            longitude = float(
                longitude
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                "Neighborhood center contains "
                "invalid coordinates."
            ) from exc

        if not (
            -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):
            raise RuntimeError(
                "Neighborhood center contains "
                "coordinates outside valid ranges."
            )

        centers_by_name[name] = {
            "name": name,
            "latitude": latitude,
            "longitude": longitude,
        }

    if (
        set(centers_by_name)
        != SARICAM_NEIGHBORHOOD_SET
    ):
        raise RuntimeError(
            "Neighborhood center data does "
            "not cover all canonical "
            "Sarıçam neighborhoods."
        )

    ordered_centers = tuple(
        centers_by_name[
            neighborhood
        ]
        for neighborhood
        in SARICAM_NEIGHBORHOODS
    )

    return (
        ordered_centers,
        centers_by_name,
    )


(
    SARICAM_NEIGHBORHOOD_CENTERS,
    SARICAM_NEIGHBORHOOD_CENTER_BY_NAME,
) = load_neighborhood_centers()