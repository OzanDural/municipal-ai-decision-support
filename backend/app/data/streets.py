import json
from pathlib import Path


STREET_DATA_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "data"
    / "locations"
    / "saricam_streets.json"
)


def load_saricam_street_data() -> dict:
    if not STREET_DATA_PATH.exists():
        raise RuntimeError(
            "Sarıçam street data file "
            f"was not found: {STREET_DATA_PATH}"
        )

    data = json.loads(
        STREET_DATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    streets = data.get(
        "streets"
    )

    if not isinstance(
        streets,
        list,
    ):
        raise RuntimeError(
            "Sarıçam street data does "
            "not contain a valid streets list."
        )

    if not streets:
        raise RuntimeError(
            "Sarıçam street list is empty."
        )

    if not all(
        isinstance(name, str)
        and name.strip()
        for name in streets
    ):
        raise RuntimeError(
            "Sarıçam street list contains "
            "invalid values."
        )

    if (
        len(streets)
        != len(set(streets))
    ):
        raise RuntimeError(
            "Sarıçam street list contains "
            "duplicate values."
        )

    declared_count = data.get(
        "count"
    )

    if (
        declared_count
        != len(streets)
    ):
        raise RuntimeError(
            "Sarıçam street count does not "
            "match the street list."
        )

    return data


SARICAM_STREET_DATA = (
    load_saricam_street_data()
)

SARICAM_STREETS = tuple(
    SARICAM_STREET_DATA[
        "streets"
    ]
)

SARICAM_STREET_SET = frozenset(
    SARICAM_STREETS
)