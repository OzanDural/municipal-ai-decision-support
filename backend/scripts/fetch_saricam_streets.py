from datetime import (
    datetime,
    timezone,
)
import json
from pathlib import Path
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.parse import urlencode
from urllib.request import (
    Request,
    urlopen,
)


OVERPASS_URLS = (
    (
        "https://overpass.private.coffee/"
        "api/interpreter"
    ),
    (
        "https://maps.mail.ru/osm/tools/"
        "overpass/api/interpreter"
    ),
    (
        "https://overpass-api.de/"
        "api/interpreter"
    ),
)

SARICAM_RELATION_ID = 1249247

OUTPUT_PATH = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "data"
    / "locations"
    / "saricam_streets.json"
)


OVERPASS_QUERY = f"""
[out:json][timeout:120];

relation({SARICAM_RELATION_ID});
map_to_area -> .searchArea;

way(area.searchArea)
  ["highway"]
  ["name"];

out tags;
"""


def normalize_name(
    value: str,
) -> str:
    return " ".join(
        value.split()
    )
ROAD_SUFFIXES = {
    "sokak": "Sokak",
    "sokağı": "Sokağı",
    "cadde": "Cadde",
    "caddesi": "Caddesi",
    "bulvar": "Bulvar",
    "bulvarı": "Bulvarı",
    "yol": "Yol",
    "yolu": "Yolu",
    "karayolu": "Karayolu",
}


def normalize_road_suffix(
    value: str,
) -> str:
    parts = value.split()

    if not parts:
        return value

    suffix_key = (
        parts[-1]
        .casefold()
        .replace("\u0307", "")
    )

    canonical_suffix = (
        ROAD_SUFFIXES.get(
            suffix_key
        )
    )

    if canonical_suffix:
        parts[-1] = (
            canonical_suffix
        )

    return " ".join(parts)


def street_dedup_key(
    value: str,
) -> str:
    return (
        value
        .casefold()
        .replace("\u0307", "")
    )
def is_supported_road_name(
    value: str,
) -> bool:
    normalized = (
        value
        .strip()
        .casefold()
    )

    return normalized.endswith(
        ALLOWED_NAME_SUFFIXES
    )

def fetch_osm_data() -> dict:
    body = urlencode(
        {
            "data":
                OVERPASS_QUERY
        }
    ).encode(
        "utf-8"
    )

    errors: list[str] = []

    for url in OVERPASS_URLS:
        print(
            "Trying Overpass endpoint:",
            url,
        )

        request = Request(
            url,
            data=body,
            headers={
                "User-Agent": (
                    "belediye-ai-karar-destek/"
                    "1.0 internship-prototype"
                ),
                "Content-Type": (
                    "application/"
                    "x-www-form-urlencoded"
                ),
            },
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=180,
            ) as response:
                raw = response.read()

            data = json.loads(
                raw.decode("utf-8")
            )

            print(
                "Overpass endpoint succeeded:",
                url,
            )

            return data

        except HTTPError as exc:
            message = (
                f"{url} -> "
                f"HTTP {exc.code}"
            )

        except URLError as exc:
            message = (
                f"{url} -> "
                f"{exc.reason}"
            )

        except TimeoutError:
            message = (
                f"{url} -> timeout"
            )

        except json.JSONDecodeError as exc:
            message = (
                f"{url} -> invalid JSON: "
                f"{exc}"
            )

        print(
            "Endpoint failed:",
            message,
        )

        errors.append(
            message
        )

    raise RuntimeError(
        "All Overpass API endpoints failed:\n"
        + "\n".join(errors)
    )

ALLOWED_HIGHWAY_TYPES = {
    "residential",
    "living_street",
    "unclassified",
    "tertiary",
    "tertiary_link",
    "secondary",
    "secondary_link",
    "primary",
    "primary_link",
    "trunk",
    "trunk_link",
}
ALLOWED_NAME_SUFFIXES = (
    "sokak",
    "sokağı",
    "cadde",
    "caddesi",
    "bulvar",
    "bulvarı",
    "yol",
    "yolu",
    "karayolu",
)
def extract_street_names(
    data: dict,
) -> list[str]:
    names_by_key: dict[
        str,
        str,
    ] = {}

    for element in data.get(
        "elements",
        [],
    ):
        tags = element.get(
            "tags",
            {},
        )

        highway_type = tags.get(
            "highway"
        )

        if (
            highway_type
            not in ALLOWED_HIGHWAY_TYPES
        ):
            continue

        name = tags.get(
            "name"
        )

        if not isinstance(
            name,
            str,
        ):
            continue

        cleaned = normalize_name(
            name
        )

        cleaned = normalize_road_suffix(
            cleaned
        )

        if not (
            cleaned
            and is_supported_road_name(
                cleaned
            )
        ):
            continue

        key = street_dedup_key(
            cleaned
        )

        if key not in names_by_key:
            names_by_key[key] = (
                cleaned
            )

    return sorted(
        names_by_key.values(),
        key=str.casefold,
    )

def main() -> None:
    print(
        "Fetching named roads "
        "inside Sarıçam..."
    )

    data = fetch_osm_data()

    streets = extract_street_names(
        data
    )

    if not streets:
        raise RuntimeError(
            "No named roads were "
            "returned by Overpass."
        )

    output = {
        "district_code":
            "SARICAM",

        "district_name":
            "Sarıçam",

        "source":
            "OpenStreetMap contributors",

        "retrieval_method":
            "Overpass API",

        "source_relation_id":
            SARICAM_RELATION_ID,

        "license":
            "ODbL 1.0",

        "retrieved_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "count":
            len(streets),

        "streets":
            streets,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print(
        "Unique named roads:",
        len(streets),
    )

    print(
        "\nFirst 20 names:"
    )

    for name in streets[:20]:
        print(
            "-",
            name,
        )


if __name__ == "__main__":
    main()