import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

SUMMARY_FILE = REPORT_DIR / (
    "03_length_distribution_summary.json"
)

RECORD_FILE = REPORT_DIR / (
    "03_length_profile.csv"
)

LONG_DEVIATION_FILE = REPORT_DIR / (
    "03_long_length_deviations.jsonl"
)


# --------------------------------------------------
# Uzun-form batch'lerde hedeflenen kelime bantları
# --------------------------------------------------

LONG_BAND_EXPECTATIONS = {
    "standard_100": {
        "25_50": 10,
        "51_80": 7,
        "81_120": 3,
    },

    "claude_50": {
        "25_50": 5,
        "51_80": 3,
        "81_120": 2,
    },
}


# --------------------------------------------------
# Yardımcılar
# --------------------------------------------------

def word_count(text):
    """
    Generation promptundaki kelime hedeflerine en yakın
    basit ve şeffaf yöntem:

    whitespace-separated token count.
    """

    return len(
        text.strip().split()
    )


def sentence_count(text):
    """
    Yaklaşık cümle sayısı.
    . ! ? ile biten parçaları temel alır.

    Bu değer yalnızca profil amaçlıdır;
    otomatik reject kriteri değildir.
    """

    text = text.strip()

    if not text:
        return 0

    parts = re.split(
        r"[.!?]+(?:\s+|$)",
        text,
    )

    parts = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    return max(
        1,
        len(parts),
    )


def length_band(count):

    if count < 25:
        return "under_25"

    if count <= 50:
        return "25_50"

    if count <= 80:
        return "51_80"

    if count <= 120:
        return "81_120"

    return "over_120"


def percentile(values, p):

    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    position = (
        (len(values) - 1)
        * (p / 100)
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(values[lower])

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def descriptive_stats(values):

    if not values:

        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    values = sorted(values)

    mean = (
        sum(values)
        / len(values)
    )

    median = percentile(
        values,
        50,
    )

    return {
        "count":
            len(values),

        "min":
            min(values),

        "mean":
            round(mean, 2),

        "median":
            round(median, 2),

        "p75":
            round(
                percentile(values, 75),
                2,
            ),

        "p90":
            round(
                percentile(values, 90),
                2,
            ),

        "p95":
            round(
                percentile(values, 95),
                2,
            ),

        "p99":
            round(
                percentile(values, 99),
                2,
            ),

        "max":
            max(values),
    }


def distribution_dict(counter):

    return {
        str(key): value
        for key, value
        in sorted(
            counter.items(),
            key=lambda x: str(x[0]),
        )
    }


# --------------------------------------------------
# Dataset oku
# --------------------------------------------------

records = []

with INPUT_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    for line_number, line in enumerate(
        f,
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        record = json.loads(
            line
        )

        batch = record[
            "generation_batch"
        ]

        source_group = (
            "long_augmentation"
            if batch.startswith("long_")
            else "original"
        )

        wc = word_count(
            record["complaint_text"]
        )

        sc = sentence_count(
            record["complaint_text"]
        )

        record_profile = {
            **record,

            "_source_group":
                source_group,

            "_word_count":
                wc,

            "_sentence_count":
                sc,

            "_length_band":
                length_band(wc),
        }

        records.append(
            record_profile
        )


# --------------------------------------------------
# Temel güvenlik
# --------------------------------------------------

if len(records) != 1750:

    raise ValueError(
        f"Beklenen 1750 kayıt, "
        f"bulunan {len(records)}."
    )


# --------------------------------------------------
# Record-level CSV
# --------------------------------------------------

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


csv_fields = [
    "sample_id",
    "generation_batch",
    "generator_id",
    "category",
    "scenario",
    "difficulty",
    "language_style",
    "has_typo",
    "location_style",
    "source_group",
    "word_count",
    "sentence_count",
    "length_band",
    "complaint_text",
]


with RECORD_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields,
    )

    writer.writeheader()

    for r in records:

        writer.writerow({
            "sample_id":
                r["sample_id"],

            "generation_batch":
                r["generation_batch"],

            "generator_id":
                r["generator_id"],

            "category":
                r["category"],

            "scenario":
                r["scenario"],

            "difficulty":
                r["difficulty"],

            "language_style":
                r["language_style"],

            "has_typo":
                r["has_typo"],

            "location_style":
                r["location_style"],

            "source_group":
                r["_source_group"],

            "word_count":
                r["_word_count"],

            "sentence_count":
                r["_sentence_count"],

            "length_band":
                r["_length_band"],

            "complaint_text":
                r["complaint_text"],
        })


# --------------------------------------------------
# Global ve source-group istatistikleri
# --------------------------------------------------

overall_word_counts = [
    r["_word_count"]
    for r in records
]

overall_sentence_counts = [
    r["_sentence_count"]
    for r in records
]


source_summary = {}


for source_group in [
    "original",
    "long_augmentation",
]:

    group_records = [
        r
        for r in records
        if r["_source_group"]
        == source_group
    ]

    wc_values = [
        r["_word_count"]
        for r in group_records
    ]

    sc_values = [
        r["_sentence_count"]
        for r in group_records
    ]

    band_counts = Counter(
        r["_length_band"]
        for r in group_records
    )

    sentence_counts = Counter(
        r["_sentence_count"]
        for r in group_records
    )

    source_summary[
        source_group
    ] = {
        "record_count":
            len(group_records),

        "word_stats":
            descriptive_stats(
                wc_values
            ),

        "sentence_stats":
            descriptive_stats(
                sc_values
            ),

        "length_band_distribution":
            distribution_dict(
                band_counts
            ),

        "sentence_count_distribution":
            distribution_dict(
                sentence_counts
            ),
    }


# --------------------------------------------------
# Generator bazında
# --------------------------------------------------

generator_summary = {}


for generator in sorted(
    {
        r["generator_id"]
        for r in records
    }
):

    group = [
        r
        for r in records
        if r["generator_id"]
        == generator
    ]

    generator_summary[
        generator
    ] = {
        "record_count":
            len(group),

        "word_stats":
            descriptive_stats([
                r["_word_count"]
                for r in group
            ]),

        "length_band_distribution":
            distribution_dict(
                Counter(
                    r["_length_band"]
                    for r in group
                )
            ),
    }


# --------------------------------------------------
# Category bazında:
# original ve long ayrı
# --------------------------------------------------

category_summary = {}


categories = sorted(
    {
        r["category"]
        for r in records
    }
)


for category in categories:

    category_summary[
        category
    ] = {}

    for source_group in [
        "original",
        "long_augmentation",
    ]:

        group = [
            r
            for r in records
            if (
                r["category"]
                == category
                and r["_source_group"]
                == source_group
            )
        ]

        category_summary[
            category
        ][source_group] = {
            "record_count":
                len(group),

            "word_stats":
                descriptive_stats([
                    r["_word_count"]
                    for r in group
                ]),

            "length_band_distribution":
                distribution_dict(
                    Counter(
                        r["_length_band"]
                        for r in group
                    )
                ),
        }


# --------------------------------------------------
# Batch bazında
# --------------------------------------------------

batch_summary = {}


batches = sorted(
    {
        r["generation_batch"]
        for r in records
    }
)


for batch in batches:

    group = [
        r
        for r in records
        if r["generation_batch"]
        == batch
    ]

    batch_summary[
        batch
    ] = {
        "record_count":
            len(group),

        "word_stats":
            descriptive_stats([
                r["_word_count"]
                for r in group
            ]),

        "length_band_distribution":
            distribution_dict(
                Counter(
                    r["_length_band"]
                    for r in group
                )
            ),

        "sentence_stats":
            descriptive_stats([
                r["_sentence_count"]
                for r in group
            ]),
    }


# --------------------------------------------------
# Long augmentation kota kontrolü
#
# Bu HARD FAIL değildir.
# "~25-50" gibi hedefler nedeniyle sonuç REVIEW adayıdır.
# --------------------------------------------------

long_quota_results = {}

long_deviations = []


long_batches = sorted(
    batch
    for batch in batches
    if batch.startswith("long_")
)


for batch in long_batches:

    batch_records = [
        r
        for r in records
        if r["generation_batch"]
        == batch
    ]

    if batch == "long_claude_001":

        expected = (
            LONG_BAND_EXPECTATIONS[
                "claude_50"
            ]
        )

    else:

        expected = (
            LONG_BAND_EXPECTATIONS[
                "standard_100"
            ]
        )


    batch_category_results = {}


    for category in categories:

        category_records = [
            r
            for r in batch_records
            if r["category"]
            == category
        ]

        band_counts = Counter(
            r["_length_band"]
            for r in category_records
        )

        quota_diffs = []


        for band, expected_count in (
            expected.items()
        ):

            actual_count = (
                band_counts.get(
                    band,
                    0,
                )
            )

            if (
                actual_count
                != expected_count
            ):

                quota_diffs.append({
                    "band":
                        band,

                    "expected":
                        expected_count,

                    "actual":
                        actual_count,
                })


        outside_target = [
            {
                "sample_id":
                    r["sample_id"],

                "word_count":
                    r["_word_count"],

                "length_band":
                    r["_length_band"],
            }
            for r in category_records
            if r["_length_band"]
            in {
                "under_25",
                "over_120",
            }
        ]


        batch_category_results[
            category
        ] = {
            "record_count":
                len(category_records),

            "band_counts":
                distribution_dict(
                    band_counts
                ),

            "quota_differences":
                quota_diffs,

            "outside_target_records":
                outside_target,
        }


        if quota_diffs or outside_target:

            deviation = {
                "generation_batch":
                    batch,

                "category":
                    category,

                "quota_differences":
                    quota_diffs,

                "outside_target_records":
                    outside_target,

                "decision":
                    "REVIEW_ONLY",

                "note":
                    (
                        "Length targets were approximate "
                        "generation constraints. "
                        "No automatic deletion or metadata "
                        "change is performed."
                    ),
            }

            long_deviations.append(
                deviation
            )


    long_quota_results[
        batch
    ] = (
        batch_category_results
    )


# --------------------------------------------------
# Uzun-form cümle yapısı
# --------------------------------------------------

long_records = [
    r
    for r in records
    if r["_source_group"]
    == "long_augmentation"
]


long_2_to_5_sentence_count = sum(
    1
    for r in long_records
    if 2 <= r["_sentence_count"] <= 5
)


long_2_to_5_ratio = (
    long_2_to_5_sentence_count
    / len(long_records)
    if long_records
    else 0
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

summary = {
    "total_records":
        len(records),

    "overall_word_stats":
        descriptive_stats(
            overall_word_counts
        ),

    "overall_sentence_stats":
        descriptive_stats(
            overall_sentence_counts
        ),

    "source_groups":
        source_summary,

    "generator_summary":
        generator_summary,

    "category_summary":
        category_summary,

    "batch_summary":
        batch_summary,

    "long_augmentation_analysis": {
        "record_count":
            len(long_records),

        "records_with_2_to_5_sentences":
            long_2_to_5_sentence_count,

        "ratio_with_2_to_5_sentences":
            round(
                long_2_to_5_ratio,
                4,
            ),

        "quota_review_group_count":
            len(long_deviations),

        "batch_category_results":
            long_quota_results,
    },

    "policy": (
        "Length statistics are descriptive QA. "
        "Approximate generation length targets are "
        "not automatic rejection criteria. "
        "BERTurk token-length analysis will be repeated "
        "after the final accepted dataset is produced."
    ),
}


with SUMMARY_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


with LONG_DEVIATION_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for item in long_deviations:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


# --------------------------------------------------
# Terminal
# --------------------------------------------------

print("=" * 78)
print("STAGE 3 - LENGTH & DISTRIBUTION QA")
print("=" * 78)

print(
    f"Toplam kayıt        : {len(records)}"
)

print(
    f"Original            : "
    f"{source_summary['original']['record_count']}"
)

print(
    f"Long augmentation   : "
    f"{source_summary['long_augmentation']['record_count']}"
)


print("\nORIGINAL kelime istatistikleri:")

original_stats = (
    source_summary[
        "original"
    ]["word_stats"]
)

for key, value in (
    original_stats.items()
):

    print(
        f"  {key:<8}: {value}"
    )


print("\nLONG kelime istatistikleri:")

long_stats = (
    source_summary[
        "long_augmentation"
    ]["word_stats"]
)

for key, value in (
    long_stats.items()
):

    print(
        f"  {key:<8}: {value}"
    )


print("\nLONG length band dağılımı:")

for band, count in (
    source_summary[
        "long_augmentation"
    ][
        "length_band_distribution"
    ].items()
):

    print(
        f"  {band:<12}: {count}"
    )


print(
    "\nLong-form 2-5 cümle oranı:"
)

print(
    f"  "
    f"{long_2_to_5_sentence_count}"
    f"/{len(long_records)} "
    f"({long_2_to_5_ratio:.2%})"
)


print(
    "\nLength quota review grubu:"
)

print(
    f"  {len(long_deviations)}"
)


print("\nDosyalar:")

print(
    f"  Summary    : {SUMMARY_FILE}"
)

print(
    f"  Record CSV : {RECORD_FILE}"
)

print(
    f"  Deviations : {LONG_DEVIATION_FILE}"
)