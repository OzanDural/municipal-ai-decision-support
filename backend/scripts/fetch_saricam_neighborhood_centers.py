import json
from pathlib import Path
import re
import sys
import time
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.parse import urlencode
from urllib.request import (
    Request,
    urlopen,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.app.data.neighborhoods import (
    SARICAM_NEIGHBORHOODS,
)


NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/"
    "search"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "locations"
    / "saricam_neighborhood_centers.json"
)


PLACE_TYPES = {
    "neighbourhood",
    "quarter",
    "suburb",
    "village",
    "hamlet",
    "locality",
}


SPECIAL_NAME_VARIANTS = {
    "Cumhuriyet (Buruk)": [
        "Cumhuriyet",
        "Buruk Cumhuriyet",
    ],
    "Büyükbaklalı": [
        "Büyükbaklalı",
        "Büyük Baklalı",
    ],
    "Küçükbaklalı": [
        "Küçükbaklalı",
        "Küçük Baklalı",
    ],
}


# Bu kayıt yalnızca haritada yaklaşık odaklama
# amacıyla kullanılır.
# Mahalle sınırı veya resmî CBS geometrisi değildir.
MANUAL_VERIFIED_CENTERS = {
    "Sarıçam": {
        "latitude": 37.152126,
        "longitude": 35.5062407,
        "osm_type": "node",
        "osm_id": 2287937701,
        "place_type": "locality",
        "address_type": "locality",
        "display_name": (
            "Sarıçam Mahallesi, "
            "Sarıçam, Adana, Türkiye"
        ),
        "resolution_method": "manual_verified",
        "needs_review": False,
        "match_score": 1000,
    },

    "Elif Su Uludağ": {
        "latitude": 37.0183523,
        "longitude": 35.4169519,
        "osm_type": None,
        "osm_id": None,
        "place_type": "neighbourhood",
        "address_type": "neighbourhood",
        "display_name": (
            "Elif Su Uludağ Mahallesi, "
            "Sarıçam, Adana, Türkiye"
        ),
        "resolution_method": "manual_verified",
        "needs_review": False,
        "match_score": 1000,
    },

    "Ertuğrulgazi": {
        "latitude": 37.0773544,
        "longitude": 35.4143917,
        "osm_type": None,
        "osm_id": None,
        "place_type": "neighbourhood",
        "address_type": "neighbourhood",
        "display_name": (
            "Ertuğrulgazi Mahallesi, "
            "Sarıçam, Adana, Türkiye"
        ),
        "resolution_method": "manual_verified",
        "needs_review": False,
        "match_score": 1000,
    },

    "Yeni Mahalle": {
        "latitude": 36.9862672,
        "longitude": 35.407684,
        "osm_type": None,
        "osm_id": None,
        "place_type": "neighbourhood",
        "address_type": "neighbourhood",
        "display_name": (
            "Yeni Mahalle, "
            "Sarıçam, Adana, Türkiye"
        ),
        "resolution_method": "manual_verified",
        "needs_review": False,
        "match_score": 1000,
    },
}
def normalize_text(
    value: str,
) -> str:
    return (
        value
        .casefold()
        .replace("\u0307", "")
        .replace("ç", "c")
        .replace("ğ", "g")
        .replace("ı", "i")
        .replace("ö", "o")
        .replace("ş", "s")
        .replace("ü", "u")
    )


def compact_text(
    value: str,
) -> str:
    normalized = normalize_text(
        value
    )

    return re.sub(
        r"[^a-z0-9]+",
        "",
        normalized,
    )


def target_variants(
    neighborhood: str,
) -> list[str]:
    special = (
        SPECIAL_NAME_VARIANTS.get(
            neighborhood
        )
    )

    if special:
        return special

    return [
        neighborhood
    ]


def text_matches_target(
    text: str,
    neighborhood: str,
) -> bool:
    candidate_key = compact_text(
        text
    )

    if not candidate_key:
        return False

    for variant in target_variants(
        neighborhood
    ):
        variant_key = compact_text(
            variant
        )

        if (
            variant_key
            and variant_key
            in candidate_key
        ):
            return True

    return False


def build_queries(
    neighborhood: str,
) -> list[str]:
    queries: list[str] = []

    for variant in target_variants(
        neighborhood
    ):
        queries.extend(
            [
                (
                    f"{variant} Mahallesi, "
                    "Sarıçam, Adana, Türkiye"
                ),
                (
                    f"{variant}, "
                    "Sarıçam, Adana, Türkiye"
                ),
            ]
        )

    if "(" in neighborhood:
        base_name = (
            neighborhood
            .split("(", 1)[0]
            .strip()
        )

        parenthetical = (
            neighborhood
            .split("(", 1)[1]
            .replace(")", "")
            .strip()
        )

        queries.extend(
            [
                (
                    f"{base_name} Mahallesi, "
                    "Sarıçam, Adana, Türkiye"
                ),
                (
                    f"{parenthetical} Mahallesi, "
                    "Sarıçam, Adana, Türkiye"
                ),
                (
                    f"{parenthetical}, "
                    "Sarıçam, Adana, Türkiye"
                ),
            ]
        )

    # Aynı sorguların tekrar gönderilmesini önler.
    unique_queries: list[str] = []
    seen: set[str] = set()

    for query in queries:
        key = normalize_text(
            query
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_queries.append(
            query
        )

    return unique_queries


def search_nominatim(
    query: str,
) -> list[dict]:
    params = urlencode(
        {
            "q": query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 5,
            "countrycodes": "tr",
        }
    )

    request = Request(
        f"{NOMINATIM_URL}?{params}",
        headers={
            "User-Agent": (
                "belediye-ai-karar-destek/"
                "1.0 internship-prototype"
            ),
            "Accept-Language":
                "tr",
        },
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:
            raw = response.read()

    except HTTPError as exc:
        raise RuntimeError(
            "Nominatim returned "
            f"HTTP {exc.code}."
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            "Could not connect to "
            "Nominatim."
        ) from exc

    return json.loads(
        raw.decode(
            "utf-8"
        )
    )


def candidate_is_in_saricam(
    candidate: dict,
) -> bool:
    display_name = normalize_text(
        str(
            candidate.get(
                "display_name",
                "",
            )
        )
    )

    return (
        "saricam" in display_name
        and "adana" in display_name
    )


def candidate_resolution_method(
    neighborhood: str,
    candidate: dict,
) -> str | None:
    if not candidate_is_in_saricam(
        candidate
    ):
        return None

    candidate_name = str(
        candidate.get(
            "name",
            "",
        )
    )

    # Kritik güvenlik kontrolü:
    # Hedef mahalle adı adayın kendi adında
    # bulunmak zorunda.
    #
    # Böylece:
    # "Türkiye İş Bankası, ..., Sarıçam"
    # yalnızca display_name içinde Sarıçam
    # geçtiği için kabul edilmez.
    if not text_matches_target(
        candidate_name,
        neighborhood,
    ):
        return None

    category = str(
        candidate.get(
            "category",
            candidate.get(
                "class",
                "",
            ),
        )
    )

    place_type = str(
        candidate.get(
            "type",
            "",
        )
    )

    address_type = str(
        candidate.get(
            "addresstype",
            "",
        )
    )

    # Nominatim'de gerçek mahalle/köy/locality
    # sonuçlarında addresstype genellikle
    # type/category bilgisinden daha güvenilirdir.
    if address_type in PLACE_TYPES:
        return "exact_place"

    if (
        category == "place"
        and place_type
        in PLACE_TYPES
    ):
        return "exact_place"

    # Örn:
    # "Dağcı Mahallesi Camii"
    # aday adı hedef mahalleyi gerçekten taşıyor
    # fakat kendisi mahalle nesnesi değil.
    return "named_poi_fallback"


def candidate_score(
    neighborhood: str,
    candidate: dict,
) -> int:
    method = (
        candidate_resolution_method(
            neighborhood,
            candidate,
        )
    )

    if method is None:
        return -1

    if method == "exact_place":
        score = 100
    else:
        score = 40

    address_type = str(
        candidate.get(
            "addresstype",
            "",
        )
    )

    place_type = str(
        candidate.get(
            "type",
            "",
        )
    )

    effective_type = (
        address_type
        if address_type
        in PLACE_TYPES
        else place_type
    )

    type_bonus = {
        "neighbourhood":
            10,
        "quarter":
            9,
        "suburb":
            8,
        "village":
            7,
        "hamlet":
            6,
        "locality":
            5,
    }

    score += type_bonus.get(
        effective_type,
        0,
    )

    candidate_name = str(
        candidate.get(
            "name",
            "",
        )
    )

    # Aday adı tam olarak hedef varyantlardan
    # biri ise küçük ek avantaj verir.
    candidate_key = compact_text(
        candidate_name
    )

    for variant in target_variants(
        neighborhood
    ):
        if (
            candidate_key
            == compact_text(
                variant
            )
        ):
            score += 5
            break

    return score


def find_neighborhood(
    neighborhood: str,
) -> dict | None:
    manual = (
        MANUAL_VERIFIED_CENTERS.get(
            neighborhood
        )
    )

    if manual is not None:
        return {
            "name":
                neighborhood,
            **manual,
        }

    best_candidate = None
    best_score = -1
    best_method = None

    for query in build_queries(
        neighborhood
    ):
        results = search_nominatim(
            query
        )

        for candidate in results:
            score = candidate_score(
                neighborhood,
                candidate,
            )

            if score > best_score:
                best_candidate = (
                    candidate
                )

                best_score = (
                    score
                )

                best_method = (
                    candidate_resolution_method(
                        neighborhood,
                        candidate,
                    )
                )

        # Gerçek mahalle/place bulunduysa
        # daha zayıf POI fallback aramaya
        # devam etmeye gerek yok.
        if (
            best_method
            == "exact_place"
        ):
            break

        time.sleep(
            1.1
        )

    if (
        best_candidate is None
        or best_score < 0
        or best_method is None
    ):
        return None

    return {
        "name":
            neighborhood,

        "latitude":
            float(
                best_candidate["lat"]
            ),

        "longitude":
            float(
                best_candidate["lon"]
            ),

        "osm_type":
            best_candidate.get(
                "osm_type"
            ),

        "osm_id":
            best_candidate.get(
                "osm_id"
            ),

        "place_type":
            best_candidate.get(
                "type"
            ),

        "address_type":
            best_candidate.get(
                "addresstype"
            ),

        "display_name":
            best_candidate.get(
                "display_name"
            ),

        "resolution_method":
            best_method,

        "needs_review":
            (
                best_method
                != "exact_place"
            ),

        "match_score":
            best_score,
    }


def main() -> None:
    centers = []
    unresolved = []

    total = len(
        SARICAM_NEIGHBORHOODS
    )

    for index, neighborhood in enumerate(
        SARICAM_NEIGHBORHOODS,
        start=1,
    ):
        print(
            f"[{index}/{total}] "
            f"{neighborhood}"
        )

        try:
            result = find_neighborhood(
                neighborhood
            )

        except RuntimeError as exc:
            print(
                "  request failed:",
                exc,
            )

            result = None

        if result is None:
            print(
                "  UNRESOLVED"
            )

            unresolved.append(
                neighborhood
            )

        else:
            print(
                "  resolved:",
                result["latitude"],
                result["longitude"],
            )

            print(
                "  method:",
                result[
                    "resolution_method"
                ],
            )

            print(
                "  source:",
                result[
                    "display_name"
                ],
            )

            centers.append(
                result
            )

        time.sleep(
            1.1
        )

    review_items = [
        item
        for item in centers
        if item.get(
            "needs_review"
        )
    ]

    exact_items = [
        item
        for item in centers
        if (
            item.get(
                "resolution_method"
            )
            == "exact_place"
        )
    ]

    manual_items = [
        item
        for item in centers
        if (
            item.get(
                "resolution_method"
            )
            == "manual_verified"
        )
    ]

    output = {
        "district_code":
            "SARICAM",

        "district_name":
            "Sarıçam",

        "source":
            "OpenStreetMap Nominatim",

        "purpose":
            (
                "Approximate map focus "
                "centers only; not "
                "neighborhood boundaries."
            ),

        "count":
            len(centers),

        "exact_place_count":
            len(exact_items),

        "manual_verified_count":
            len(manual_items),

        "review_required_count":
            len(review_items),

        "unresolved_count":
            len(unresolved),

        "centers":
            centers,

        "unresolved":
            unresolved,
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

    print()
    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print(
        "Resolved:",
        len(centers),
    )

    print(
        "Exact place:",
        len(exact_items),
    )

    print(
        "Manual verified:",
        len(manual_items),
    )

    print(
        "Needs manual review:",
        len(review_items),
    )

    print(
        "Unresolved:",
        len(unresolved),
    )

    if review_items:
        print()
        print(
            "Needs manual review:"
        )

        for item in review_items:
            print(
                "-",
                item["name"],
                "->",
                item[
                    "resolution_method"
                ],
                "->",
                item[
                    "display_name"
                ],
            )

    if unresolved:
        print()
        print(
            "Unresolved neighborhoods:"
        )

        for name in unresolved:
            print(
                "-",
                name,
            )


if __name__ == "__main__":
    main()