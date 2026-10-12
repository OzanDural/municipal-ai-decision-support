from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from geoalchemy2.elements import WKTElement
from sqlalchemy import delete, func, select, text

import backend.app.main as main_module
from backend.app.db.database import SessionLocal
from backend.app.models.complaint import Complaint
from backend.app.models.enums import (
    ComplaintCategory,
    ComplaintStatus,
    UserRole,
)
from backend.app.models.user import User
from backend.app.services.service_area import (
    ServiceAreaNotConfiguredError,
    is_location_within_service_area,
)


# ============================================================
# AMAÇ
# ============================================================
#
# Bu script:
# - gerçek vatandaş verisi üretmez,
# - gerçek belediye trendi iddia etmez,
# - sentetik / küratörlü metinleri demo verisi olarak kullanır,
# - AMA AI tahmin alanlarını doğrudan uygulamanın GERÇEK
#   fine-tuned BERTurk classifier'ından alır.
#
# Tarih ve konumlar kontrollü demo senaryosu için sentetiktir.
# İnsan review sonuçları seed sırasında üretilmez.
# Review gereken kayıtlar Operator Panelindeki gerçek kullanıcı
# kararı için REVIEW_REQUIRED olarak bırakılır.
#
# AI alanları ASLA simüle edilmez:
# - predicted_category
# - secondary_category
# - model_score
# - second_model_score
# - score_margin
# - model_version
# - review_required
# - review_reason
#
# ============================================================


DEMO_PREFIX = "SRM-DEMO-"
DEFAULT_COUNT = 150
DEFAULT_RANDOM_SEED = 20260913

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "final"
    / "test.jsonl"
)

CHALLENGE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "challenge"
    / "challenge_set_final_v1.jsonl"
)

CENTERS_PATH = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "locations"
    / "saricam_neighborhood_centers.json"
)


# 150 kayıt için yaklaşık:
# 30  -> son 7 gün
# 40  -> 8-30 gün
# 50  -> 31-90 gün
# 30  -> 91-180 gün
DATE_BUCKETS = {
    "0_6": {
        "ratio": 0.20,
        "min_days": 0,
        "max_days": 6,
    },
    "7_29": {
        "ratio": 0.2666666667,
        "min_days": 7,
        "max_days": 29,
    },
    "30_89": {
        "ratio": 0.3333333333,
        "min_days": 30,
        "max_days": 89,
    },
    "90_179": {
        "ratio": 0.20,
        "min_days": 90,
        "max_days": 179,
    },
}


# Zaman dilimlerinde aynı kategori dağılımını kullanmıyoruz.
# Böylece dashboard / LLM gerçek model çıktıları üzerinde,
# fakat kontrollü bir DEMO senaryosunda zaman trendi görebilir.
#
# Bunlar AI tahminlerini değiştirmez.
# Yalnızca hangi gold-label metinlerin hangi tarihe atanacağını
# belirler.
CATEGORY_WEIGHTS_BY_BUCKET: dict[
    str,
    list[tuple[ComplaintCategory, int]],
] = {
    "0_6": [
        (ComplaintCategory.TEMIZLIK_ATIK, 34),
        (ComplaintCategory.YOL_KALDIRIM, 20),
        (ComplaintCategory.PARK_YESIL_ALAN, 12),
        (ComplaintCategory.KENT_DUZENI_DENETIM, 21),
        (ComplaintCategory.HAYVAN_VETERINER, 13),
    ],
    "7_29": [
        (ComplaintCategory.TEMIZLIK_ATIK, 29),
        (ComplaintCategory.YOL_KALDIRIM, 25),
        (ComplaintCategory.PARK_YESIL_ALAN, 14),
        (ComplaintCategory.KENT_DUZENI_DENETIM, 18),
        (ComplaintCategory.HAYVAN_VETERINER, 14),
    ],
    "30_89": [
        (ComplaintCategory.TEMIZLIK_ATIK, 21),
        (ComplaintCategory.YOL_KALDIRIM, 27),
        (ComplaintCategory.PARK_YESIL_ALAN, 22),
        (ComplaintCategory.KENT_DUZENI_DENETIM, 16),
        (ComplaintCategory.HAYVAN_VETERINER, 14),
    ],
    "90_179": [
        (ComplaintCategory.TEMIZLIK_ATIK, 19),
        (ComplaintCategory.YOL_KALDIRIM, 31),
        (ComplaintCategory.PARK_YESIL_ALAN, 15),
        (ComplaintCategory.KENT_DUZENI_DENETIM, 14),
        (ComplaintCategory.HAYVAN_VETERINER, 21),
    ],
}


# Challenge kayıtlarını özellikle son dönemde biraz daha fazla
# kullanıyoruz. Böylece son dönem ekranında gerçek modelin
# zor örneklere verdiği uncertainty davranışı da görülebilir.
CHALLENGE_SHARE_BY_BUCKET = {
    "0_6": 0.40,
    "7_29": 0.32,
    "30_89": 0.24,
    "90_179": 0.18,
}


# Konumsal demo analizi için bazı kategori-mahalle kümeleri.
# Noktalar yine gerçek Sarıçam service-area kontrolünden geçer.
CATEGORY_HOTSPOTS: dict[
    ComplaintCategory,
    list[str],
] = {
    ComplaintCategory.TEMIZLIK_ATIK: [
        "Kemalpaşa",
        "Yeşiltepe",
        "Yeni Mahalle",
        "Yavuz Sultan Selim",
    ],
    ComplaintCategory.YOL_KALDIRIM: [
        "Yavuz Sultan Selim",
        "Yıldırım Beyazıt",
        "Yeniyayla",
        "Kemalpaşa",
    ],
    ComplaintCategory.PARK_YESIL_ALAN: [
        "Yeşiltepe",
        "Yavuz Sultan Selim",
        "Kemalpaşa",
        "Yeni Mahalle",
    ],
    ComplaintCategory.KENT_DUZENI_DENETIM: [
        "Kemalpaşa",
        "Yıldırım Beyazıt",
        "Yeni Mahalle",
        "Yeşiltepe",
    ],
    ComplaintCategory.HAYVAN_VETERINER: [
        "Yürekli",
        "Yeniyayla",
        "Yeni Mahalle",
        "Yarımca",
    ],
}


ADDRESS_CONTEXTS = [
    "Haritada işaretlenen yerleşim bölgesi.",
    "Mahalle içindeki ortak kullanım alanına yakın bölüm.",
    "Konutların bulunduğu yerleşim çevresi.",
    "Mahalle merkezine yakın harita noktası.",
    "Yaya kullanımının yoğun olduğu yerleşim bölümü.",
    "Mahalle içindeki yol bağlantısı çevresi.",
]


@dataclass(frozen=True)
class SourceRecord:
    complaint_text: str
    gold_category: ComplaintCategory
    source: str
    source_id: str | None = None
    challenge_type: str | None = None


@dataclass(frozen=True)
class NeighborhoodCenter:
    neighborhood: str
    latitude: float
    longitude: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manager dashboard için gerçek BERTurk inference "
            "çıktıları kullanan sentetik tarih/konum demo verisi."
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=(
            "Oluşturulacak demo kayıt sayısı. "
            "Varsayılan: 150"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Tekrarlanabilir demo rastgelelik tohumu.",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Gerçek BERTurk inference yapar ve dağılımı gösterir, "
            "ancak veritabanına commit etmez."
        ),
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Yalnızca SRM-DEMO- ile başlayan demo kayıtlarını siler."
        ),
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Mevcut SRM-DEMO- kayıtlarını silip yeniden üretir."
        ),
    )

    parser.add_argument(
        "--allow-test-db",
        action="store_true",
        help=(
            "Adında 'test' geçen veritabanında çalışmaya açıkça izin verir."
        ),
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"Required JSONL file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_no, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    (
                        f"Invalid JSON in {path} "
                        f"at line {line_no}: {exc}"
                    )
                ) from exc

            if isinstance(
                item,
                dict,
            ):
                records.append(
                    item
                )

    return records


def normalize_text(
    value: str,
) -> str:
    return " ".join(
        value
        .strip()
        .lower()
        .split()
    )


def load_source_records() -> list[SourceRecord]:
    result: list[SourceRecord] = []

    # --------------------------------------------------------
    # 1) Frozen standard TEST split
    # --------------------------------------------------------
    #
    # Buradaki kayıtlar modelin eğitim amacıyla kullanıldığı
    # training splitinden gelmez. Dashboard demo analitiği için
    # daha savunulabilir bir kaynak oluşturur.
    # --------------------------------------------------------

    for item in load_jsonl(
        TEST_DATA_PATH
    ):
        complaint_text = str(
            item.get(
                "complaint_text",
                "",
            )
        ).strip()

        category_value = (
            item.get("category")
            or item.get("gold_category")
        )

        if (
            not complaint_text
            or category_value is None
        ):
            continue

        try:
            gold_category = (
                ComplaintCategory(
                    str(
                        category_value
                    )
                )
            )
        except ValueError:
            continue

        result.append(
            SourceRecord(
                complaint_text=complaint_text,
                gold_category=gold_category,
                source="TEST",
                source_id=(
                    str(
                        item.get(
                            "sample_id"
                        )
                    )
                    if item.get(
                        "sample_id"
                    )
                    is not None
                    else None
                ),
            )
        )

    # --------------------------------------------------------
    # 2) Frozen challenge set - yalnız IN_SCOPE kayıtlar
    # --------------------------------------------------------
    #
    # OOD kayıtları closed-set belediye şikâyet tablosuna
    # bilerek koymuyoruz.
    # --------------------------------------------------------

    for item in load_jsonl(
        CHALLENGE_DATA_PATH
    ):
        if (
            item.get(
                "challenge_group"
            )
            != "IN_SCOPE"
        ):
            continue

        complaint_text = str(
            item.get(
                "complaint_text",
                "",
            )
        ).strip()

        category_value = (
            item.get(
                "gold_category"
            )
        )

        if (
            not complaint_text
            or category_value is None
        ):
            continue

        try:
            gold_category = (
                ComplaintCategory(
                    str(
                        category_value
                    )
                )
            )
        except ValueError:
            continue

        result.append(
            SourceRecord(
                complaint_text=complaint_text,
                gold_category=gold_category,
                source="CHALLENGE",
                source_id=(
                    str(
                        item.get(
                            "challenge_id"
                        )
                    )
                    if item.get(
                        "challenge_id"
                    )
                    is not None
                    else None
                ),
                challenge_type=(
                    str(
                        item.get(
                            "challenge_type"
                        )
                    )
                    if item.get(
                        "challenge_type"
                    )
                    is not None
                    else None
                ),
            )
        )

    # Source'lar arasında normalize edilmiş duplicate varsa
    # yalnız ilkini tut.
    unique: dict[
        str,
        SourceRecord,
    ] = {}

    for record in result:
        unique.setdefault(
            normalize_text(
                record.complaint_text
            ),
            record,
        )

    records = list(
        unique.values()
    )

    if not records:
        raise SystemExit(
            "No usable demo source records were found."
        )

    return records


def first_value(
    item: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        if key in item:
            return item[key]

    return None


def load_neighborhood_centers() -> list[NeighborhoodCenter]:
    if not CENTERS_PATH.exists():
        raise SystemExit(
            (
                "Neighborhood centers file not found: "
                f"{CENTERS_PATH}"
            )
        )

    raw = json.loads(
        CENTERS_PATH.read_text(
            encoding="utf-8"
        )
    )

    items: list[
        dict[str, Any]
    ] = []

    if isinstance(
        raw,
        list,
    ):
        items = [
            item
            for item in raw
            if isinstance(
                item,
                dict,
            )
        ]

    elif isinstance(
        raw,
        dict,
    ):
        for container_key in (
            "centers",
            "neighborhoods",
            "results",
            "items",
        ):
            candidate = (
                raw.get(
                    container_key
                )
            )

            if isinstance(
                candidate,
                list,
            ):
                items = [
                    item
                    for item in candidate
                    if isinstance(
                        item,
                        dict,
                    )
                ]

                break

        if not items:
            for key, value in (
                raw.items()
            ):
                if not isinstance(
                    value,
                    dict,
                ):
                    continue

                item = dict(
                    value
                )

                if not first_value(
                    item,
                    (
                        "neighborhood",
                        "name",
                        "mahalle",
                    ),
                ):
                    item[
                        "neighborhood"
                    ] = key

                items.append(
                    item
                )

    centers: list[
        NeighborhoodCenter
    ] = []

    for item in items:
        neighborhood = first_value(
            item,
            (
                "neighborhood",
                "name",
                "mahalle",
            ),
        )

        latitude = first_value(
            item,
            (
                "latitude",
                "lat",
            ),
        )

        longitude = first_value(
            item,
            (
                "longitude",
                "lon",
                "lng",
            ),
        )

        if (
            not neighborhood
            or latitude is None
            or longitude is None
        ):
            continue

        try:
            centers.append(
                NeighborhoodCenter(
                    neighborhood=(
                        str(
                            neighborhood
                        ).strip()
                    ),
                    latitude=float(
                        latitude
                    ),
                    longitude=float(
                        longitude
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    unique: dict[
        str,
        NeighborhoodCenter,
    ] = {}

    for center in centers:
        unique.setdefault(
            center.neighborhood,
            center,
        )

    result = list(
        unique.values()
    )

    if not result:
        raise SystemExit(
            (
                "Neighborhood centers file exists "
                "but no usable coordinates were parsed."
            )
        )

    return result


def allocate_counts(
    total: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    if total < 1:
        raise SystemExit(
            "--count must be at least 1."
        )

    raw_counts = {
        key:
            total * ratio
        for key, ratio
        in ratios.items()
    }

    counts = {
        key:
            int(value)
        for key, value
        in raw_counts.items()
    }

    remainder = (
        total
        - sum(
            counts.values()
        )
    )

    fractional_order = sorted(
        ratios.keys(),
        key=lambda key: (
            raw_counts[key]
            - counts[key]
        ),
        reverse=True,
    )

    for key in (
        fractional_order[
            :remainder
        ]
    ):
        counts[key] += 1

    return counts


def weighted_choice(
    rng: random.Random,
    weighted_values: list[
        tuple[Any, int]
    ],
):
    values = [
        value
        for value, _
        in weighted_values
    ]

    weights = [
        weight
        for _, weight
        in weighted_values
    ]

    return rng.choices(
        values,
        weights=weights,
        k=1,
    )[0]


def build_slots(
    rng: random.Random,
    count: int,
) -> list[
    tuple[
        str,
        ComplaintCategory,
        str,
    ]
]:
    bucket_counts = allocate_counts(
        count,
        {
            key:
                float(
                    config[
                        "ratio"
                    ]
                )
            for key, config
            in DATE_BUCKETS.items()
        },
    )

    slots: list[
        tuple[
            str,
            ComplaintCategory,
            str,
        ]
    ] = []

    for (
        bucket,
        bucket_count,
    ) in bucket_counts.items():
        challenge_share = (
            CHALLENGE_SHARE_BY_BUCKET[
                bucket
            ]
        )

        for _ in range(
            bucket_count
        ):
            category = weighted_choice(
                rng,
                CATEGORY_WEIGHTS_BY_BUCKET[
                    bucket
                ],
            )

            source = (
                "CHALLENGE"
                if rng.random()
                < challenge_share
                else "TEST"
            )

            slots.append(
                (
                    bucket,
                    category,
                    source,
                )
            )

    rng.shuffle(
        slots
    )

    return slots


def build_source_pools(
    records: list[SourceRecord],
    rng: random.Random,
) -> dict[
    tuple[
        str,
        ComplaintCategory,
    ],
    list[SourceRecord],
]:
    pools: dict[
        tuple[
            str,
            ComplaintCategory,
        ],
        list[SourceRecord],
    ] = defaultdict(
        list
    )

    for record in records:
        pools[
            (
                record.source,
                record.gold_category,
            )
        ].append(
            record
        )

    for values in (
        pools.values()
    ):
        rng.shuffle(
            values
        )

    return pools


def take_source_record(
    pools: dict[
        tuple[
            str,
            ComplaintCategory,
        ],
        list[SourceRecord],
    ],
    preferred_source: str,
    category: ComplaintCategory,
) -> SourceRecord:
    preferred_key = (
        preferred_source,
        category,
    )

    if pools.get(
        preferred_key
    ):
        return pools[
            preferred_key
        ].pop()

    fallback_source = (
        "TEST"
        if preferred_source
        == "CHALLENGE"
        else "CHALLENGE"
    )

    fallback_key = (
        fallback_source,
        category,
    )

    if pools.get(
        fallback_key
    ):
        return pools[
            fallback_key
        ].pop()

    raise SystemExit(
        (
            "Not enough source records for "
            f"category={category.value}. "
            "Reduce --count."
        )
    )


def choose_center(
    rng: random.Random,
    centers: list[NeighborhoodCenter],
    category: ComplaintCategory,
) -> NeighborhoodCenter:
    by_name = {
        center.neighborhood:
            center
        for center in centers
    }

    preferred = [
        by_name[name]
        for name
        in CATEGORY_HOTSPOTS[
            category
        ]
        if name in by_name
    ]

    if (
        preferred
        and rng.random()
        < 0.72
    ):
        return rng.choice(
            preferred
        )

    return rng.choice(
        centers
    )


def choose_valid_point(
    db,
    rng: random.Random,
    center: NeighborhoodCenter,
) -> tuple[
    float,
    float,
]:
    for _ in range(
        24
    ):
        latitude = (
            center.latitude
            + rng.uniform(
                -0.0016,
                0.0016,
            )
        )

        longitude = (
            center.longitude
            + rng.uniform(
                -0.0018,
                0.0018,
            )
        )

        if is_location_within_service_area(
            db,
            latitude=latitude,
            longitude=longitude,
        ):
            return (
                latitude,
                longitude,
            )

    if is_location_within_service_area(
        db,
        latitude=center.latitude,
        longitude=center.longitude,
    ):
        return (
            center.latitude,
            center.longitude,
        )

    raise ValueError(
        (
            "No valid service-area point "
            f"for {center.neighborhood}."
        )
    )


def get_valid_centers(
    db,
    centers: list[NeighborhoodCenter],
) -> list[NeighborhoodCenter]:
    valid: list[
        NeighborhoodCenter
    ] = []

    try:
        for center in centers:
            if is_location_within_service_area(
                db,
                latitude=center.latitude,
                longitude=center.longitude,
            ):
                valid.append(
                    center
                )

    except ServiceAreaNotConfiguredError as exc:
        raise SystemExit(
            (
                "Sarıçam service area is not configured "
                "in the selected database."
            )
        ) from exc

    if not valid:
        raise SystemExit(
            (
                "No neighborhood center is inside "
                "the configured Sarıçam service area."
            )
        )

    return valid


def get_operator_id(
    db,
) -> int:
    operator = db.execute(
        select(
            User
        )
        .where(
            User.role
            == UserRole.OPERATOR
        )
        .where(
            User.is_active
            == True
        )
        .order_by(
            User.id.asc()
        )
        .limit(1)
    ).scalar_one_or_none()

    if operator is None:
        raise SystemExit(
            (
                "No active OPERATOR user found. "
                "Create one before seeding historical "
                "reviewed demo records."
            )
        )

    return operator.id


def demo_count(
    db,
) -> int:
    return int(
        db.execute(
            select(
                func.count(
                    Complaint.id
                )
            )
            .where(
                Complaint.application_no.like(
                    f"{DEMO_PREFIX}%"
                )
            )
        ).scalar_one()
        or 0
    )


def clear_demo_rows(
    db,
) -> int:
    existing = demo_count(
        db
    )

    if existing == 0:
        return 0

    db.execute(
        delete(
            Complaint
        )
        .where(
            Complaint.application_no.like(
                f"{DEMO_PREFIX}%"
            )
        )
    )

    return existing


def age_days_for_bucket(
    rng: random.Random,
    bucket: str,
) -> int:
    config = DATE_BUCKETS[
        bucket
    ]

    return rng.randint(
        int(
            config[
                "min_days"
            ]
        ),
        int(
            config[
                "max_days"
            ]
        ),
    )


def validate_real_prediction(
    prediction: dict[str, Any],
) -> None:
    required_fields = {
        "predicted_category",
        "secondary_category",
        "model_score",
        "second_model_score",
        "score_margin",
        "model_version",
        "review_required",
        "review_reason",
    }

    missing = (
        required_fields
        - set(
            prediction
        )
    )

    if missing:
        raise RuntimeError(
            (
                "Classifier prediction is missing fields: "
                f"{sorted(missing)}"
            )
        )

    model_version = str(
        prediction[
            "model_version"
        ]
    )

    # Pytest FakeClassifier gibi test doubles ile
    # yanlışlıkla demo DB üretmeyi engelle.
    if (
        not model_version
        or model_version
        == "test-model"
        or "fake" in model_version.lower()
    ):
        raise RuntimeError(
            (
                "Safety stop: classifier does not look like "
                "the real application model. "
                f"model_version={model_version!r}"
            )
        )

    ComplaintCategory(
        str(
            prediction[
                "predicted_category"
            ]
        )
    )

    ComplaintCategory(
        str(
            prediction[
                "secondary_category"
            ]
        )
    )

    for field in (
        "model_score",
        "second_model_score",
        "score_margin",
    ):
        value = float(
            prediction[
                field
            ]
        )

        if not (
            0.0
            <= value
            <= 1.0
        ):
            raise RuntimeError(
                (
                    "Classifier returned an invalid "
                    f"{field}: {value}"
                )
            )


def build_historical_workflow(
    *,
    rng: random.Random,
    bucket: str,
    created_at: datetime,
    gold_category: ComplaintCategory,
    predicted_category: ComplaintCategory,
    model_review_required: bool,
    operator_id: int,
    now: datetime,
) -> dict[str, Any]:
    # --------------------------------------------------------
    # Nihai demo politikası
    # --------------------------------------------------------
    #
    # Seed script insan kararı üretmez.
    #
    # Gerçek fine-tuned BERTurk:
    #   review_required=False -> AUTO_ACCEPTED
    #   review_required=True  -> REVIEW_REQUIRED
    #
    # REVIEW_REQUIRED kayıtların final_category,
    # was_corrected, reviewed_by ve reviewed_at alanları
    # yalnız Operator Panelindeki gerçek kullanıcı işlemiyle
    # doldurulur.
    #
    # Böylece Manager Dashboard ve ilerideki LLM analitiği,
    # AI-insan karşılaştırmasında gerçek operatör kararlarını
    # kullanır.
    # --------------------------------------------------------

    if not model_review_required:
        return {
            "status":
                ComplaintStatus.AUTO_ACCEPTED,

            "review_required":
                False,

            "final_category":
                None,

            "was_corrected":
                None,

            "reviewed_by":
                None,

            "reviewed_at":
                None,

            "review_origin":
                "NONE",
        }

    return {
        "status":
            ComplaintStatus.REVIEW_REQUIRED,

        "review_required":
            True,

        "final_category":
            None,

        "was_corrected":
            None,

        "reviewed_by":
            None,

        "reviewed_at":
            None,

        "review_origin":
            "MODEL_REQUIRED_PENDING",
    }


def print_profile(
    records: list[Complaint],
    metadata_rows: list[
        dict[str, Any]
    ],
) -> None:
    source_counts = Counter(
        row[
            "source"
        ]
        for row in metadata_rows
    )

    bucket_counts = Counter(
        row[
            "bucket"
        ]
        for row in metadata_rows
    )

    gold_counts = Counter(
        row[
            "gold_category"
        ]
        for row in metadata_rows
    )

    prediction_counts = Counter(
        record
        .predicted_category
        .value
        for record in records
    )

    effective_counts = Counter(
        (
            record
            .final_category
            .value
            if record
            .final_category
            is not None
            else record
            .predicted_category
            .value
        )
        for record
        in records
    )

    status_counts = Counter(
        record
        .status
        .value
        for record in records
    )

    model_versions = Counter(
        record.model_version
        for record in records
    )

    review_origins = Counter(
        row[
            "historical_review_origin"
        ]
        for row in metadata_rows
    )

    initial_review_required = sum(
        1
        for row in metadata_rows
        if row[
            "model_review_required"
        ]
    )

    initial_review_rate = (
        initial_review_required
        / len(records)
        if records
        else 0.0
    )

    model_review_reasons = Counter(
        (
            row[
                "model_review_reason"
            ]
            or "NONE"
        )
        for row in metadata_rows
        if row[
            "model_review_required"
        ]
    )

    model_correct = sum(
        1
        for row in metadata_rows
        if row[
            "predicted_category"
        ]
        == row[
            "gold_category"
        ]
    )

    reviewed = sum(
        1
        for record in records
        if record.status
        == ComplaintStatus.REVIEWED
    )

    corrected = sum(
        1
        for record in records
        if record.was_corrected
        is True
    )

    neighborhoods = Counter(
        record.neighborhood
        for record in records
    )

    print()
    print("=" * 72)
    print("REAL_BERTURK_DEMO_PROFILE")
    print("=" * 72)

    print(
        f"total={len(records)}"
    )

    print()
    print("sources=")

    for (
        key,
        value,
    ) in sorted(
        source_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print("date_buckets=")

    for key in (
        "0_6",
        "7_29",
        "30_89",
        "90_179",
    ):
        print(
            f"  {key}: "
            f"{bucket_counts[key]}"
        )

    print()
    print("gold_categories=")

    for (
        key,
        value,
    ) in sorted(
        gold_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print("real_model_predictions=")

    for (
        key,
        value,
    ) in sorted(
        prediction_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print("effective_categories_after_review=")

    for (
        key,
        value,
    ) in sorted(
        effective_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print("final_statuses=")

    for (
        key,
        value,
    ) in sorted(
        status_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print("model_versions=")

    for (
        key,
        value,
    ) in sorted(
        model_versions.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print("historical_review_origins=")

    for (
        key,
        value,
    ) in sorted(
        review_origins.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "real_model_initial_review_required="
        f"{initial_review_required}"
    )

    print(
        "real_model_initial_review_rate="
        f"{initial_review_rate:.4f}"
    )

    print()
    print("real_model_review_reasons=")

    for (
        key,
        value,
    ) in sorted(
        model_review_reasons.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "real_model_vs_gold_correct="
        f"{model_correct}/{len(records)}"
    )

    print(
        "synthetic_historical_reviewed="
        f"{reviewed}"
    )

    print(
        "synthetic_historical_corrected="
        f"{corrected}"
    )

    print()
    print("top_neighborhoods=")

    for (
        neighborhood,
        count,
    ) in neighborhoods.most_common(
        12
    ):
        print(
            f"  {neighborhood}: {count}"
        )

    print()
    print(
        "IMPORTANT: model prediction fields above are "
        "REAL application BERTurk outputs."
    )

    print(
        "IMPORTANT: dates, locations and historical human-review "
        "timing are SYNTHETIC DEMO context."
    )

    print("=" * 72)


def generate_records(
    db,
    classifier,
    *,
    count: int,
    seed: int,
) -> tuple[
    list[Complaint],
    list[dict[str, Any]],
]:
    rng = random.Random(
        seed
    )

    source_records = (
        load_source_records()
    )

    pools = build_source_pools(
        source_records,
        rng,
    )

    slots = build_slots(
        rng,
        count,
    )

    centers = (
        get_valid_centers(
            db,
            load_neighborhood_centers(),
        )
    )

    operator_id = (
        get_operator_id(
            db
        )
    )

    now = datetime.now(
        timezone.utc
    )

    records: list[
        Complaint
    ] = []

    metadata_rows: list[
        dict[str, Any]
    ] = []

    for (
        index,
        (
            bucket,
            requested_category,
            preferred_source,
        ),
    ) in enumerate(
        slots,
        start=1,
    ):
        source_record = (
            take_source_record(
                pools,
                preferred_source,
                requested_category,
            )
        )

        # ====================================================
        # GERÇEK MODEL INFERENCE
        # ====================================================
        prediction = classifier.predict(
            source_record.complaint_text
        )

        validate_real_prediction(
            prediction
        )

        predicted_category = (
            ComplaintCategory(
                str(
                    prediction[
                        "predicted_category"
                    ]
                )
            )
        )

        secondary_category = (
            ComplaintCategory(
                str(
                    prediction[
                        "secondary_category"
                    ]
                )
            )
        )

        model_review_required = bool(
            prediction[
                "review_required"
            ]
        )

        # ====================================================
        # SENTETİK TARİH
        # ====================================================

        age_days = (
            age_days_for_bucket(
                rng,
                bucket,
            )
        )

        created_at = (
            now
            - timedelta(
                days=age_days,
                hours=rng.randint(
                    0,
                    22,
                ),
                minutes=rng.randint(
                    0,
                    59,
                ),
            )
        )

        # ====================================================
        # SENTETİK KONUM, GERÇEK SERVICE-AREA KONTROLÜ
        # ====================================================

        center = choose_center(
            rng,
            centers,
            source_record.gold_category,
        )

        try:
            (
                latitude,
                longitude,
            ) = choose_valid_point(
                db,
                rng,
                center,
            )
        except ValueError:
            # Nadir durumda farklı doğrulanmış merkeze geç.
            center = rng.choice(
                centers
            )

            (
                latitude,
                longitude,
            ) = choose_valid_point(
                db,
                rng,
                center,
            )

        # ====================================================
        # TARİHSEL İNSAN REVIEW DEMOSU
        # ====================================================

        workflow = (
            build_historical_workflow(
                rng=rng,
                bucket=bucket,
                created_at=created_at,
                gold_category=(
                    source_record.gold_category
                ),
                predicted_category=(
                    predicted_category
                ),
                model_review_required=(
                    model_review_required
                ),
                operator_id=operator_id,
                now=now,
            )
        )

        updated_at = (
            workflow[
                "reviewed_at"
            ]
            if workflow[
                "reviewed_at"
            ]
            is not None
            else created_at
        )

        address_text = (
            f"{center.neighborhood}: "
            + rng.choice(
                ADDRESS_CONTEXTS
            )
        )

        complaint = Complaint(
            application_no=(
                f"{DEMO_PREFIX}"
                f"{index:06d}"
            ),

            complaint_text=(
                source_record
                .complaint_text
            ),

            neighborhood=(
                center.neighborhood
            ),

            street_name=None,

            address_text=(
                address_text
            ),

            location=WKTElement(
                (
                    f"POINT("
                    f"{longitude} "
                    f"{latitude}"
                    f")"
                ),
                srid=4326,
            ),

            # --------------------------------------------
            # GERÇEK BERTURK OUTPUT
            # --------------------------------------------

            predicted_category=(
                predicted_category
            ),

            secondary_category=(
                secondary_category
            ),

            model_score=float(
                prediction[
                    "model_score"
                ]
            ),

            second_model_score=float(
                prediction[
                    "second_model_score"
                ]
            ),

            score_margin=float(
                prediction[
                    "score_margin"
                ]
            ),

            model_version=str(
                prediction[
                    "model_version"
                ]
            ),

            # review_reason modelin orijinal gerekçesidir.
            review_reason=(
                (
                    str(
                        prediction[
                            "review_reason"
                        ]
                    )
                )
                if prediction[
                    "review_reason"
                ]
                is not None
                else None
            ),

            # --------------------------------------------
            # TARİHSEL WORKFLOW DURUMU
            # --------------------------------------------

            review_required=(
                workflow[
                    "review_required"
                ]
            ),

            final_category=(
                workflow[
                    "final_category"
                ]
            ),

            was_corrected=(
                workflow[
                    "was_corrected"
                ]
            ),

            reviewed_by=(
                workflow[
                    "reviewed_by"
                ]
            ),

            reviewed_at=(
                workflow[
                    "reviewed_at"
                ]
            ),

            status=(
                workflow[
                    "status"
                ]
            ),

            created_at=(
                created_at
            ),

            updated_at=(
                updated_at
            ),
        )

        records.append(
            complaint
        )

        metadata_rows.append(
            {
                "source":
                    source_record.source,

                "source_id":
                    source_record.source_id,

                "challenge_type":
                    source_record.challenge_type,

                "bucket":
                    bucket,

                "gold_category":
                    source_record
                    .gold_category
                    .value,

                "predicted_category":
                    predicted_category
                    .value,

                "model_review_required":
                    model_review_required,

                "model_review_reason":
                    (
                        str(
                            prediction[
                                "review_reason"
                            ]
                        )
                        if prediction[
                            "review_reason"
                        ]
                        is not None
                        else None
                    ),

                "historical_review_origin":
                    workflow[
                        "review_origin"
                    ],

                "model_version":
                    str(
                        prediction[
                            "model_version"
                        ]
                    ),
            }
        )

        if (
            index == 1
            or index % 10 == 0
            or index == count
        ):
            print(
                (
                    "REAL_BERTURK_INFERENCE "
                    f"{index}/{count}"
                )
            )

    return (
        records,
        metadata_rows,
    )


def main() -> None:
    args = parse_args()

    if (
        args.clear
        and args.replace
    ):
        raise SystemExit(
            (
                "Use either --clear or --replace, "
                "not both."
            )
        )

    if args.count < 1:
        raise SystemExit(
            "--count must be at least 1."
        )

    db = SessionLocal()

    try:
        database_name = (
            db.execute(
                text(
                    "SELECT current_database()"
                )
            )
            .scalar_one()
        )

        print(
            f"DATABASE={database_name}"
        )

        if (
            "test"
            in str(
                database_name
            ).lower()
            and not args.allow_test_db
        ):
            raise SystemExit(
                (
                    "Safety stop: selected database name "
                    "contains 'test'. If this is intentional, "
                    "use --allow-test-db."
                )
            )

        # ----------------------------------------------------
        # Sadece demo verilerini sil.
        # Model yüklemeye gerek yok.
        # ----------------------------------------------------
        if args.clear:
            deleted = (
                clear_demo_rows(
                    db
                )
            )

            db.commit()

            print(
                "DEMO_ROWS_DELETED="
                f"{deleted}"
            )

            return

        existing = (
            demo_count(
                db
            )
        )

        if (
            existing > 0
            and not args.replace
        ):
            raise SystemExit(
                (
                    f"{existing} demo rows already exist. "
                    "Use --replace to rebuild them or "
                    "--clear to delete only demo rows."
                )
            )

        if args.replace:
            deleted = (
                clear_demo_rows(
                    db
                )
            )

            print(
                "DEMO_ROWS_MARKED_FOR_REPLACEMENT="
                f"{deleted}"
            )

        print(
            "Loading application lifespan and REAL BERTurk classifier..."
        )

        # ----------------------------------------------------
        # FastAPI uygulamasının kendi lifespan'ını kullanıyoruz.
        # Böylece seed script ayrı bir model-serving mantığı
        # yazmıyor; uygulama hangi classifier'ı kullanıyorsa
        # aynısını kullanıyor.
        # ----------------------------------------------------

        with TestClient(
            main_module.app
        ):
            classifier = (
                main_module
                .app
                .state
                .classifier
            )

            print(
                (
                    "Classifier loaded: "
                    f"{classifier.__class__.__name__}"
                )
            )

            (
                records,
                metadata_rows,
            ) = generate_records(
                db,
                classifier,
                count=args.count,
                seed=args.seed,
            )

        db.add_all(
            records
        )

        # DB constraint/FK/PostGIS hatalarını commit öncesinde
        # görmek için flush.
        db.flush()

        print_profile(
            records,
            metadata_rows,
        )

        if args.preview:
            db.rollback()

            print()
            print(
                "PREVIEW_ONLY: no demo rows were committed."
            )

            return

        db.commit()

        print()
        print(
            "DEMO_ROWS_INSERTED="
            f"{len(records)}"
        )

        print(
            "DEMO_PREFIX="
            f"{DEMO_PREFIX}"
        )

        print(
            (
                "AI_VALUES="
                "REAL_APPLICATION_BERTURK_OUTPUTS"
            )
        )

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
