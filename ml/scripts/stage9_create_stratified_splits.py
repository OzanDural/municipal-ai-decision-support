import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


FINAL_FILE = Path(
    "data/processed/master_final.jsonl"
)

TOKEN_PROFILE_FILE = Path(
    "ml/reports/dataset_qa/"
    "08b_final_length_tokenizer_profile.csv"
)

SPLIT_DIR = Path(
    "data/splits/provisional"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

TRAIN_FILE = SPLIT_DIR / "train.jsonl"
VAL_FILE = SPLIT_DIR / "validation.jsonl"
TEST_FILE = SPLIT_DIR / "test.jsonl"

MANIFEST_FILE = REPORT_DIR / (
    "09_split_manifest.csv"
)

SUMMARY_FILE = REPORT_DIR / (
    "09_split_summary.json"
)


EXPECTED_FINAL_SHA256 = (
    "c6d8a06e0fb757c3b1db11a87959f7086fd4b22da5c97268522c5fd4d77ed1c7"
)

RANDOM_SEED = 2026

TRAIN_SIZE = 1167
VAL_SIZE = 250
TEST_SIZE = 250

EXPECTED_TOTAL = (
    TRAIN_SIZE
    + VAL_SIZE
    + TEST_SIZE
)


# ==================================================
# Helpers
# ==================================================

def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def word_band(n):

    if n <= 24:
        return "UNDER_25"

    if n <= 50:
        return "25_50"

    if n <= 80:
        return "51_80"

    if n <= 120:
        return "81_120"

    return "OVER_120"


def percentile_stats(values):

    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "min":
            float(np.min(arr)),

        "mean":
            float(np.mean(arr)),

        "median":
            float(np.median(arr)),

        "p95":
            float(
                np.percentile(
                    arr,
                    95,
                )
            ),

        "p99":
            float(
                np.percentile(
                    arr,
                    99,
                )
            ),

        "max":
            float(np.max(arr)),
    }


def counter_dict(values):

    return dict(
        sorted(
            Counter(values).items()
        )
    )


def stratum_key(record):

    return (
        record["category"]
        + "||"
        + record["scenario"]
    )


def split_statistics(
    split_records,
    profile_map,
):

    sample_ids = [
        r["sample_id"]
        for r in split_records
    ]

    word_counts = [
        int(
            profile_map[sid][
                "word_count"
            ]
        )
        for sid in sample_ids
    ]

    token_counts = [
        int(
            profile_map[sid][
                "berturk_token_count"
            ]
        )
        for sid in sample_ids
    ]

    return {
        "count":
            len(split_records),

        "category_distribution":
            counter_dict([
                r["category"]
                for r in split_records
            ]),

        "category_scenario_distribution":
            counter_dict([
                stratum_key(r)
                for r in split_records
            ]),

        "generator_distribution":
            counter_dict([
                r["generator_id"]
                for r in split_records
            ]),

        "difficulty_distribution":
            counter_dict([
                r["difficulty"]
                for r in split_records
            ]),

        "source_family_distribution":
            counter_dict([
                profile_map[
                    r["sample_id"]
                ][
                    "source_family"
                ]
                for r in split_records
            ]),

        "word_band_distribution":
            counter_dict([
                word_band(
                    int(
                        profile_map[
                            r["sample_id"]
                        ][
                            "word_count"
                        ]
                    )
                )
                for r in split_records
            ]),

        "word_count":
            percentile_stats(
                word_counts
            ),

        "berturk_token_count":
            percentile_stats(
                token_counts
            ),
    }


def max_distribution_deviation_pp(
    overall_records,
    split_records,
    key_function,
):

    overall_counter = Counter(
        key_function(r)
        for r in overall_records
    )

    split_counter = Counter(
        key_function(r)
        for r in split_records
    )

    overall_n = len(
        overall_records
    )

    split_n = len(
        split_records
    )

    deviations = []

    for key in overall_counter:

        overall_ratio = (
            overall_counter[key]
            / overall_n
        )

        split_ratio = (
            split_counter.get(
                key,
                0,
            )
            / split_n
        )

        deviations.append(
            abs(
                split_ratio
                - overall_ratio
            )
            * 100
        )

    return float(
        max(deviations)
    )


# ==================================================
# Verify frozen final dataset
# ==================================================

actual_final_hash = sha256_file(
    FINAL_FILE
)


if (
    actual_final_hash
    != EXPECTED_FINAL_SHA256
):

    raise ValueError(
        "master_final.jsonl hash değişmiş!\n"
        f"Beklenen : {EXPECTED_FINAL_SHA256}\n"
        f"Bulunan  : {actual_final_hash}\n"
        "Split oluşturulmadı."
    )


# ==================================================
# Load final dataset
# ==================================================

records = []

with FINAL_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        records.append(
            json.loads(line)
        )


if len(records) != EXPECTED_TOTAL:

    raise ValueError(
        f"Beklenen final kayıt "
        f"{EXPECTED_TOTAL}, "
        f"bulunan {len(records)}."
    )


sample_ids = [
    r["sample_id"]
    for r in records
]


if len(
    set(sample_ids)
) != EXPECTED_TOTAL:

    raise ValueError(
        "Final dataset içinde "
        "duplicate sample_id bulundu."
    )


if any(
    r["qa_status"]
    != "ACCEPTED"
    for r in records
):

    raise ValueError(
        "Final dataset içinde ACCEPTED "
        "olmayan qa_status bulundu."
    )


# ==================================================
# Load Stage 8B tokenizer/length profile
# ==================================================

profile_map = {}


with TOKEN_PROFILE_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    required_fields = {
        "sample_id",
        "source_family",
        "word_count",
        "berturk_token_count",
    }

    if not reader.fieldnames:

        raise ValueError(
            "Tokenizer profile header yok."
        )

    missing = (
        required_fields
        - set(reader.fieldnames)
    )

    if missing:

        raise ValueError(
            "Tokenizer profile eksik kolonlar: "
            f"{sorted(missing)}"
        )

    for row in reader:

        sample_id = row[
            "sample_id"
        ]

        if sample_id in profile_map:

            raise ValueError(
                "Tokenizer profile duplicate "
                f"sample_id: {sample_id}"
            )

        profile_map[
            sample_id
        ] = row


if len(profile_map) != EXPECTED_TOTAL:

    raise ValueError(
        "Tokenizer profile row sayısı "
        f"beklenenden farklı: "
        f"{len(profile_map)}"
    )


if set(profile_map) != set(
    sample_ids
):

    raise ValueError(
        "Tokenizer profile ve final "
        "dataset sample_id setleri "
        "eşleşmiyor."
    )


# ==================================================
# Build stratification labels
# ==================================================

strata = np.asarray([
    stratum_key(record)
    for record in records
])


stratum_counts = Counter(
    strata
)


print(
    f"Category+scenario stratum : "
    f"{len(stratum_counts)}"
)

print(
    f"Minimum stratum size      : "
    f"{min(stratum_counts.values())}"
)

print(
    f"Maximum stratum size      : "
    f"{max(stratum_counts.values())}"
)


# 25 strata bekliyoruz:
# 5 category x 5 scenario.
if len(stratum_counts) != 25:

    raise ValueError(
        f"Beklenen 25 category+scenario "
        f"stratum, bulunan "
        f"{len(stratum_counts)}."
    )


# İkinci stratified split için de
# yeterince temsil bulunduğunu garanti et.
if min(
    stratum_counts.values()
) < 10:

    raise ValueError(
        "En az bir category+scenario "
        "stratum güvenilir üçlü "
        "stratification için çok küçük."
    )


indices = np.arange(
    len(records)
)


# ==================================================
# Split 1:
# TRAIN = 1167
# TEMP  = 500
#
# TEMP daha sonra:
# VAL  = 250
# TEST = 250
# ==================================================

first_split = StratifiedShuffleSplit(
    n_splits=1,
    train_size=TRAIN_SIZE,
    test_size=(
        VAL_SIZE
        + TEST_SIZE
    ),
    random_state=RANDOM_SEED,
)


train_idx, temp_idx = next(
    first_split.split(
        indices,
        strata,
    )
)


if len(train_idx) != TRAIN_SIZE:

    raise ValueError(
        "Train size hatalı."
    )


if len(temp_idx) != (
    VAL_SIZE
    + TEST_SIZE
):

    raise ValueError(
        "Temp size hatalı."
    )


# ==================================================
# Split 2:
# TEMP -> VALIDATION + TEST
# ==================================================

temp_strata = strata[
    temp_idx
]


second_split = StratifiedShuffleSplit(
    n_splits=1,
    train_size=VAL_SIZE,
    test_size=TEST_SIZE,
    random_state=(
        RANDOM_SEED
        + 1
    ),
)


val_local_idx, test_local_idx = next(
    second_split.split(
        np.arange(
            len(temp_idx)
        ),
        temp_strata,
    )
)


val_idx = temp_idx[
    val_local_idx
]

test_idx = temp_idx[
    test_local_idx
]


if len(val_idx) != VAL_SIZE:

    raise ValueError(
        "Validation size hatalı."
    )


if len(test_idx) != TEST_SIZE:

    raise ValueError(
        "Test size hatalı."
    )


# ==================================================
# Preserve original final-dataset order
# inside each split.
# ==================================================

train_idx = sorted(
    train_idx.tolist()
)

val_idx = sorted(
    val_idx.tolist()
)

test_idx = sorted(
    test_idx.tolist()
)


train_records = [
    records[i]
    for i in train_idx
]

val_records = [
    records[i]
    for i in val_idx
]

test_records = [
    records[i]
    for i in test_idx
]


# ==================================================
# Conservation / overlap
# ==================================================

train_ids = {
    r["sample_id"]
    for r in train_records
}

val_ids = {
    r["sample_id"]
    for r in val_records
}

test_ids = {
    r["sample_id"]
    for r in test_records
}


if (
    train_ids
    & val_ids
):

    raise ValueError(
        "TRAIN / VAL overlap var."
    )


if (
    train_ids
    & test_ids
):

    raise ValueError(
        "TRAIN / TEST overlap var."
    )


if (
    val_ids
    & test_ids
):

    raise ValueError(
        "VAL / TEST overlap var."
    )


if (
    train_ids
    | val_ids
    | test_ids
) != set(
    sample_ids
):

    raise ValueError(
        "Split conservation başarısız."
    )


if (
    len(train_records)
    + len(val_records)
    + len(test_records)
    != EXPECTED_TOTAL
):

    raise ValueError(
        "Split record conservation "
        "başarısız."
    )


# ==================================================
# Create directories
# ==================================================

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# Write provisional split files
# ==================================================

for path, split_records in [
    (
        TRAIN_FILE,
        train_records,
    ),
    (
        VAL_FILE,
        val_records,
    ),
    (
        TEST_FILE,
        test_records,
    ),
]:

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in split_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ==================================================
# Split assignment map
# ==================================================

split_map = {}


for sid in train_ids:
    split_map[sid] = "TRAIN"

for sid in val_ids:
    split_map[sid] = "VALIDATION"

for sid in test_ids:
    split_map[sid] = "TEST"


# ==================================================
# Manifest
# ==================================================

manifest_rows = []


for original_order, record in enumerate(
    records,
    start=1,
):

    sid = record[
        "sample_id"
    ]

    profile = profile_map[
        sid
    ]

    manifest_rows.append({
        "original_order":
            original_order,

        "sample_id":
            sid,

        "split":
            split_map[sid],

        "category":
            record["category"],

        "scenario":
            record["scenario"],

        "difficulty":
            record["difficulty"],

        "language_style":
            record["language_style"],

        "has_typo":
            record["has_typo"],

        "generator_id":
            record["generator_id"],

        "generation_batch":
            record["generation_batch"],

        "source_family":
            profile[
                "source_family"
            ],

        "word_count":
            int(
                profile[
                    "word_count"
                ]
            ),

        "berturk_token_count":
            int(
                profile[
                    "berturk_token_count"
                ]
            ),
    })


manifest_fields = [
    "original_order",
    "sample_id",
    "split",
    "category",
    "scenario",
    "difficulty",
    "language_style",
    "has_typo",
    "generator_id",
    "generation_batch",
    "source_family",
    "word_count",
    "berturk_token_count",
]


with MANIFEST_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=manifest_fields,
    )

    writer.writeheader()

    writer.writerows(
        manifest_rows
    )


# ==================================================
# Statistics
# ==================================================

overall_stats = split_statistics(
    records,
    profile_map,
)

train_stats = split_statistics(
    train_records,
    profile_map,
)

val_stats = split_statistics(
    val_records,
    profile_map,
)

test_stats = split_statistics(
    test_records,
    profile_map,
)


def deviation_summary(
    split_records,
):

    return {
        "category_max_abs_deviation_pp":
            max_distribution_deviation_pp(
                records,
                split_records,
                lambda r:
                    r["category"],
            ),

        "category_scenario_max_abs_deviation_pp":
            max_distribution_deviation_pp(
                records,
                split_records,
                stratum_key,
            ),

        "generator_max_abs_deviation_pp":
            max_distribution_deviation_pp(
                records,
                split_records,
                lambda r:
                    r["generator_id"],
            ),

        "difficulty_max_abs_deviation_pp":
            max_distribution_deviation_pp(
                records,
                split_records,
                lambda r:
                    r["difficulty"],
            ),

        "source_family_max_abs_deviation_pp":
            max_distribution_deviation_pp(
                records,
                split_records,
                lambda r:
                    profile_map[
                        r["sample_id"]
                    ][
                        "source_family"
                    ],
            ),
    }


deviations = {
    "TRAIN":
        deviation_summary(
            train_records
        ),

    "VALIDATION":
        deviation_summary(
            val_records
        ),

    "TEST":
        deviation_summary(
            test_records
        ),
}


# ==================================================
# Hashes
#
# Bunlar PROVISIONAL.
# Stage 10 leakage audit bitmeden
# TEST frozen sayılmayacak.
# ==================================================

train_hash = sha256_file(
    TRAIN_FILE
)

val_hash = sha256_file(
    VAL_FILE
)

test_hash = sha256_file(
    TEST_FILE
)


# ==================================================
# Summary
# ==================================================

summary = {
    "status":
        "PROVISIONAL_STRATIFIED_SPLIT_COMPLETE",

    "input_dataset":
        str(FINAL_FILE),

    "input_record_count":
        EXPECTED_TOTAL,

    "input_dataset_sha256":
        actual_final_hash,

    "random_seed":
        RANDOM_SEED,

    "split_strategy":
        "STRATIFIED_SHUFFLE_SPLIT",

    "primary_stratification":
        "category + scenario",

    "train_count":
        len(train_records),

    "validation_count":
        len(val_records),

    "test_count":
        len(test_records),

    "train_ratio":
        len(train_records)
        / EXPECTED_TOTAL,

    "validation_ratio":
        len(val_records)
        / EXPECTED_TOTAL,

    "test_ratio":
        len(test_records)
        / EXPECTED_TOTAL,

    "stratum_count":
        len(stratum_counts),

    "minimum_stratum_size":
        min(
            stratum_counts.values()
        ),

    "maximum_stratum_size":
        max(
            stratum_counts.values()
        ),

    "overlap_checks": {
        "train_validation":
            len(
                train_ids
                & val_ids
            ),

        "train_test":
            len(
                train_ids
                & test_ids
            ),

        "validation_test":
            len(
                val_ids
                & test_ids
            ),
    },

    "record_conservation_pass":
        True,

    "overall_statistics":
        overall_stats,

    "train_statistics":
        train_stats,

    "validation_statistics":
        val_stats,

    "test_statistics":
        test_stats,

    "distribution_deviation_pp":
        deviations,

    "provisional_hashes": {
        "train":
            train_hash,

        "validation":
            val_hash,

        "test":
            test_hash,
    },

    "test_frozen":
        False,

    "stage10_leakage_audit_required":
        True,

    "policy": (
        "The frozen final accepted dataset was "
        "split deterministically using category + "
        "scenario stratification. Generator, "
        "difficulty, source-family and length "
        "distributions were retained as diagnostics "
        "rather than hard split constraints. "
        "These split files are provisional until "
        "cross-split exact, lexical and semantic "
        "leakage auditing is completed. The test "
        "set must not be used for model selection "
        "or hyperparameter tuning."
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


# ==================================================
# Terminal
# ==================================================

print("=" * 78)
print("STAGE 9 - PROVISIONAL STRATIFIED TRAIN / VAL / TEST SPLIT")
print("=" * 78)

print(
    f"Input final dataset     : "
    f"{len(records)}"
)

print(
    f"Input SHA256            : "
    f"{actual_final_hash}"
)

print(
    f"Random seed             : "
    f"{RANDOM_SEED}"
)

print(
    f"Stratification          : "
    f"category + scenario"
)


print("\nSplit sizes:")

print(
    f"  TRAIN                 : "
    f"{len(train_records)} "
    f"({len(train_records) / EXPECTED_TOTAL * 100:.2f}%)"
)

print(
    f"  VALIDATION            : "
    f"{len(val_records)} "
    f"({len(val_records) / EXPECTED_TOTAL * 100:.2f}%)"
)

print(
    f"  TEST                  : "
    f"{len(test_records)} "
    f"({len(test_records) / EXPECTED_TOTAL * 100:.2f}%)"
)


print("\nOverlap:")

print(
    f"  TRAIN / VAL           : "
    f"{len(train_ids & val_ids)}"
)

print(
    f"  TRAIN / TEST          : "
    f"{len(train_ids & test_ids)}"
)

print(
    f"  VAL / TEST            : "
    f"{len(val_ids & test_ids)}"
)

print(
    f"  Conservation          : "
    f"{len(train_records)} + "
    f"{len(val_records)} + "
    f"{len(test_records)} = "
    f"{EXPECTED_TOTAL}"
)


print("\nCategory distributions:")

for category in sorted(
    overall_stats[
        "category_distribution"
    ]
):

    print(
        f"  {category:<24} "
        f"ALL={overall_stats['category_distribution'].get(category, 0):>3} "
        f"TRAIN={train_stats['category_distribution'].get(category, 0):>3} "
        f"VAL={val_stats['category_distribution'].get(category, 0):>3} "
        f"TEST={test_stats['category_distribution'].get(category, 0):>3}"
    )


print("\nSource family:")

for family in [
    "ORIGINAL",
    "LONG",
]:

    print(
        f"  {family:<10} "
        f"ALL={overall_stats['source_family_distribution'].get(family, 0):>4} "
        f"TRAIN={train_stats['source_family_distribution'].get(family, 0):>4} "
        f"VAL={val_stats['source_family_distribution'].get(family, 0):>3} "
        f"TEST={test_stats['source_family_distribution'].get(family, 0):>3}"
    )


print("\nBERTurk token means:")

print(
    f"  ALL                   : "
    f"{overall_stats['berturk_token_count']['mean']:.2f}"
)

print(
    f"  TRAIN                 : "
    f"{train_stats['berturk_token_count']['mean']:.2f}"
)

print(
    f"  VALIDATION            : "
    f"{val_stats['berturk_token_count']['mean']:.2f}"
)

print(
    f"  TEST                  : "
    f"{test_stats['berturk_token_count']['mean']:.2f}"
)


print("\nMaximum distribution deviation (percentage points):")

for split_name in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    d = deviations[
        split_name
    ]

    print(
        f"  {split_name:<10} "
        f"category="
        f"{d['category_max_abs_deviation_pp']:.3f}, "
        f"cat+scenario="
        f"{d['category_scenario_max_abs_deviation_pp']:.3f}, "
        f"generator="
        f"{d['generator_max_abs_deviation_pp']:.3f}, "
        f"difficulty="
        f"{d['difficulty_max_abs_deviation_pp']:.3f}, "
        f"source="
        f"{d['source_family_max_abs_deviation_pp']:.3f}"
    )


print("\nPROVISIONAL hashes:")

print(
    f"  TRAIN                 : "
    f"{train_hash}"
)

print(
    f"  VALIDATION            : "
    f"{val_hash}"
)

print(
    f"  TEST                  : "
    f"{test_hash}"
)


print("\nOutputs:")

print(
    f"  Train    : {TRAIN_FILE}"
)

print(
    f"  Val      : {VAL_FILE}"
)

print(
    f"  Test     : {TEST_FILE}"
)

print(
    f"  Manifest : {MANIFEST_FILE}"
)

print(
    f"  Summary  : {SUMMARY_FILE}"
)


print(
    "\nIMPORTANT: TEST IS NOT FROZEN YET."
)

print(
    "NEXT: Stage 10 cross-split "
    "leakage audit."
)

print(
    "\nSTATUS: "
    "PROVISIONAL_STRATIFIED_SPLIT_COMPLETE"
)