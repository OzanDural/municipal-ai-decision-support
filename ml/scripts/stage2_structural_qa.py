import json
from collections import Counter, defaultdict
from pathlib import Path


INPUT_FILE = Path("data/raw/master_raw.jsonl")

SUMMARY_FILE = Path(
    "ml/reports/dataset_qa/02_structural_summary.json"
)

ERROR_FILE = Path(
    "ml/reports/dataset_qa/02_structural_errors.jsonl"
)


EXPECTED_FIELDS = {
    "sample_id",
    "complaint_text",
    "category",
    "scenario",
    "difficulty",
    "language_style",
    "has_typo",
    "location_style",
    "generator_id",
    "generation_batch",
    "qa_status",
}


CATEGORY_SCENARIOS = {
    "TEMIZLIK_ATIK": {
        "overflowing_container",
        "uncollected_garbage",
        "street_litter",
        "illegal_dumping",
        "bulky_waste_debris",
    },

    "YOL_KALDIRIM": {
        "pothole",
        "asphalt_damage",
        "sidewalk_damage",
        "curb_damage",
        "road_surface_collapse",
    },

    "PARK_YESIL_ALAN": {
        "playground_equipment",
        "park_furniture",
        "vegetation_irrigation",
        "tree_branch_hazard",
        "park_fence_path",
    },

    "KENT_DUZENI_DENETIM": {
        "business_sidewalk_occupation",
        "unauthorized_street_vendor",
        "public_space_obstruction",
        "business_noise_nuisance",
        "unauthorized_signage_display",
    },

    "HAYVAN_VETERINER": {
        "injured_stray",
        "sick_stray",
        "aggressive_stray",
        "stray_animal_group",
        "trapped_animal_rescue",
    },
}


ALLOWED_DIFFICULTY = {
    "easy",
    "medium",
    "hard",
}

ALLOWED_LANGUAGE_STYLE = {
    "formal",
    "neutral",
    "casual",
}

ALLOWED_LOCATION_STYLE = {
    "none",
    "generic",
    "specific",
}

ALLOWED_GENERATORS = {
    "GPT",
    "GEMINI",
    "CLAUDE",
}


# --------------------------------------------------
# Beklenen batch yapısı
# --------------------------------------------------

BATCH_CONFIG = {}


# Eski GPT/Gemini üretimleri
for generator_lower, generator_upper in [
    ("gpt", "GPT"),
    ("gemini", "GEMINI"),
]:

    for i in range(1, 6):

        batch = f"prod_{generator_lower}_{i:03d}"

        BATCH_CONFIG[batch] = {
            "generator": generator_upper,
            "count": 100,
            "per_category": 20,
            "per_scenario": 4,
            "difficulty_per_category": {
                "easy": 6,
                "medium": 10,
                "hard": 4,
            },
            "style_per_category": {
                "formal": 6,
                "neutral": 8,
                "casual": 6,
            },
            "typo_per_category": {
                False: 16,
                True: 4,
            },
            "location_per_category": {
                "none": 6,
                "generic": 8,
                "specific": 6,
            },
        }


# Eski Claude: 2 × 50
for i in range(1, 3):

    batch = f"prod_claude_{i:03d}"

    BATCH_CONFIG[batch] = {
        "generator": "CLAUDE",
        "count": 50,
        "per_category": 10,
        "per_scenario": 2,
        "difficulty_per_category": {
            "easy": 3,
            "medium": 5,
            "hard": 2,
        },
        "style_per_category": {
            "formal": 3,
            "neutral": 4,
            "casual": 3,
        },
        "typo_per_category": {
            False: 8,
            True: 2,
        },
        "location_per_category": {
            "none": 3,
            "generic": 4,
            "specific": 3,
        },
    }


# Yeni uzun GPT/Gemini: 3 × 100
for generator_lower, generator_upper in [
    ("gpt", "GPT"),
    ("gemini", "GEMINI"),
]:

    for i in range(1, 4):

        batch = f"long_{generator_lower}_{i:03d}"

        BATCH_CONFIG[batch] = {
            "generator": generator_upper,
            "count": 100,
            "per_category": 20,
            "per_scenario": 4,

            # Uzun-form promptunda özellikle
            # 4 easy / 10 medium / 6 hard istemiştik.
            "difficulty_per_category": {
                "easy": 4,
                "medium": 10,
                "hard": 6,
            },

            "style_per_category": {
                "formal": 6,
                "neutral": 8,
                "casual": 6,
            },

            "typo_per_category": {
                False: 16,
                True: 4,
            },

            "location_per_category": {
                "none": 6,
                "generic": 8,
                "specific": 6,
            },
        }


# Yeni uzun Claude: 1 × 50
BATCH_CONFIG["long_claude_001"] = {
    "generator": "CLAUDE",
    "count": 50,
    "per_category": 10,
    "per_scenario": 2,

    "difficulty_per_category": {
        "easy": 3,
        "medium": 5,
        "hard": 2,
    },

    "style_per_category": {
        "formal": 3,
        "neutral": 4,
        "casual": 3,
    },

    "typo_per_category": {
        False: 8,
        True: 2,
    },

    "location_per_category": {
        "none": 3,
        "generic": 4,
        "specific": 3,
    },
}


# --------------------------------------------------
# Yardımcı
# --------------------------------------------------

def add_error(errors, record, error_type, detail):

    errors.append({
        "sample_id": record.get("sample_id"),
        "generation_batch": record.get("generation_batch"),
        "error_type": error_type,
        "detail": detail,
    })


# --------------------------------------------------
# JSONL oku
# --------------------------------------------------

records = []

json_errors = []


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

        try:
            obj = json.loads(line)

        except json.JSONDecodeError as e:

            json_errors.append({
                "line_number": line_number,
                "error": str(e),
            })

            continue

        records.append(obj)


print("=" * 78)
print("STAGE 2 - STRUCTURAL + BATCH QUOTA QA")
print("=" * 78)

print(f"Okunan kayıt: {len(records)}")
print(f"JSON hatası  : {len(json_errors)}")


errors = []


# --------------------------------------------------
# Record-level structural kontroller
# --------------------------------------------------

for record in records:

    actual_fields = set(
        record.keys()
    )

    if actual_fields != EXPECTED_FIELDS:

        missing = sorted(
            EXPECTED_FIELDS - actual_fields
        )

        extra = sorted(
            actual_fields - EXPECTED_FIELDS
        )

        add_error(
            errors,
            record,
            "SCHEMA_FIELDS",
            {
                "missing": missing,
                "extra": extra,
            },
        )


    # complaint_text
    complaint_text = record.get(
        "complaint_text"
    )

    if (
        not isinstance(complaint_text, str)
        or not complaint_text.strip()
    ):
        add_error(
            errors,
            record,
            "INVALID_COMPLAINT_TEXT",
            complaint_text,
        )


    # category
    category = record.get(
        "category"
    )

    if category not in CATEGORY_SCENARIOS:

        add_error(
            errors,
            record,
            "INVALID_CATEGORY",
            category,
        )


    # scenario
    scenario = record.get(
        "scenario"
    )

    if (
        category in CATEGORY_SCENARIOS
        and scenario
        not in CATEGORY_SCENARIOS[category]
    ):

        add_error(
            errors,
            record,
            "CATEGORY_SCENARIO_MISMATCH",
            {
                "category": category,
                "scenario": scenario,
            },
        )


    # difficulty
    if record.get(
        "difficulty"
    ) not in ALLOWED_DIFFICULTY:

        add_error(
            errors,
            record,
            "INVALID_DIFFICULTY",
            record.get("difficulty"),
        )


    # language style
    if record.get(
        "language_style"
    ) not in ALLOWED_LANGUAGE_STYLE:

        add_error(
            errors,
            record,
            "INVALID_LANGUAGE_STYLE",
            record.get("language_style"),
        )


    # typo must really be bool
    if type(
        record.get("has_typo")
    ) is not bool:

        add_error(
            errors,
            record,
            "INVALID_HAS_TYPO",
            record.get("has_typo"),
        )


    # location
    if record.get(
        "location_style"
    ) not in ALLOWED_LOCATION_STYLE:

        add_error(
            errors,
            record,
            "INVALID_LOCATION_STYLE",
            record.get("location_style"),
        )


    # generator
    generator = record.get(
        "generator_id"
    )

    if generator not in ALLOWED_GENERATORS:

        add_error(
            errors,
            record,
            "INVALID_GENERATOR",
            generator,
        )


    # batch
    batch = record.get(
        "generation_batch"
    )

    if batch not in BATCH_CONFIG:

        add_error(
            errors,
            record,
            "UNKNOWN_BATCH",
            batch,
        )

    else:

        expected_generator = (
            BATCH_CONFIG[batch]["generator"]
        )

        if generator != expected_generator:

            add_error(
                errors,
                record,
                "GENERATOR_BATCH_MISMATCH",
                {
                    "generator_id": generator,
                    "generation_batch": batch,
                    "expected_generator":
                        expected_generator,
                },
            )


    # qa_status
    if record.get(
        "qa_status"
    ) != "PENDING":

        add_error(
            errors,
            record,
            "INVALID_QA_STATUS",
            record.get("qa_status"),
        )


    # sample_id prefix
    sample_id = record.get(
        "sample_id"
    )

    if not isinstance(
        sample_id,
        str,
    ):

        add_error(
            errors,
            record,
            "INVALID_SAMPLE_ID",
            sample_id,
        )

    elif (
        generator in ALLOWED_GENERATORS
        and batch in BATCH_CONFIG
    ):

        expected_prefix = (
            f"{generator}_{batch}_"
        )

        if not sample_id.startswith(
            expected_prefix
        ):

            add_error(
                errors,
                record,
                "SAMPLE_ID_PREFIX_MISMATCH",
                {
                    "sample_id": sample_id,
                    "expected_prefix":
                        expected_prefix,
                },
            )

        else:

            suffix = sample_id[
                len(expected_prefix):
            ]

            if (
                len(suffix) != 3
                or not suffix.isdigit()
            ):

                add_error(
                    errors,
                    record,
                    "INVALID_SAMPLE_ID_SEQUENCE",
                    sample_id,
                )


# --------------------------------------------------
# Unique ID
# --------------------------------------------------

id_counts = Counter(
    r.get("sample_id")
    for r in records
)

duplicate_ids = {
    sample_id: count
    for sample_id, count
    in id_counts.items()
    if count > 1
}


for sample_id, count in duplicate_ids.items():

    errors.append({
        "sample_id": sample_id,
        "generation_batch": None,
        "error_type":
            "DUPLICATE_SAMPLE_ID",
        "detail": {
            "count": count,
        },
    })


# --------------------------------------------------
# Batch-level quota kontrolü
# --------------------------------------------------

records_by_batch = defaultdict(
    list
)

for record in records:

    records_by_batch[
        record.get("generation_batch")
    ].append(record)


batch_results = {}


for batch, config in sorted(
    BATCH_CONFIG.items()
):

    batch_records = records_by_batch.get(
        batch,
        [],
    )

    batch_errors = []


    # toplam batch sayısı
    if len(batch_records) != config["count"]:

        batch_errors.append({
            "type": "BATCH_COUNT",
            "expected": config["count"],
            "actual": len(batch_records),
        })


    category_counts = Counter(
        r.get("category")
        for r in batch_records
    )


    for category in CATEGORY_SCENARIOS:

        actual = category_counts.get(
            category,
            0,
        )

        expected = config[
            "per_category"
        ]

        if actual != expected:

            batch_errors.append({
                "type":
                    "CATEGORY_COUNT",

                "category":
                    category,

                "expected":
                    expected,

                "actual":
                    actual,
            })


        category_records = [
            r
            for r in batch_records
            if r.get("category")
            == category
        ]


        # scenario
        scenario_counts = Counter(
            r.get("scenario")
            for r in category_records
        )

        for scenario in sorted(
            CATEGORY_SCENARIOS[category]
        ):

            actual_scenario = (
                scenario_counts.get(
                    scenario,
                    0,
                )
            )

            expected_scenario = (
                config["per_scenario"]
            )

            if (
                actual_scenario
                != expected_scenario
            ):

                batch_errors.append({
                    "type":
                        "SCENARIO_COUNT",

                    "category":
                        category,

                    "scenario":
                        scenario,

                    "expected":
                        expected_scenario,

                    "actual":
                        actual_scenario,
                })


        # difficulty
        difficulty_counts = Counter(
            r.get("difficulty")
            for r in category_records
        )

        for value, expected in (
            config[
                "difficulty_per_category"
            ].items()
        ):

            actual = difficulty_counts.get(
                value,
                0,
            )

            if actual != expected:

                batch_errors.append({
                    "type":
                        "DIFFICULTY_COUNT",

                    "category":
                        category,

                    "value":
                        value,

                    "expected":
                        expected,

                    "actual":
                        actual,
                })


        # style
        style_counts = Counter(
            r.get("language_style")
            for r in category_records
        )

        for value, expected in (
            config[
                "style_per_category"
            ].items()
        ):

            actual = style_counts.get(
                value,
                0,
            )

            if actual != expected:

                batch_errors.append({
                    "type":
                        "STYLE_COUNT",

                    "category":
                        category,

                    "value":
                        value,

                    "expected":
                        expected,

                    "actual":
                        actual,
                })


        # typo
        typo_counts = Counter(
            r.get("has_typo")
            for r in category_records
        )

        for value, expected in (
            config[
                "typo_per_category"
            ].items()
        ):

            actual = typo_counts.get(
                value,
                0,
            )

            if actual != expected:

                batch_errors.append({
                    "type":
                        "TYPO_COUNT",

                    "category":
                        category,

                    "value":
                        value,

                    "expected":
                        expected,

                    "actual":
                        actual,
                })


        # location
        location_counts = Counter(
            r.get("location_style")
            for r in category_records
        )

        for value, expected in (
            config[
                "location_per_category"
            ].items()
        ):

            actual = (
                location_counts.get(
                    value,
                    0,
                )
            )

            if actual != expected:

                batch_errors.append({
                    "type":
                        "LOCATION_COUNT",

                    "category":
                        category,

                    "value":
                        value,

                    "expected":
                        expected,

                    "actual":
                        actual,
                })


    batch_results[batch] = {
        "record_count":
            len(batch_records),

        "category_counts":
            dict(category_counts),

        "quota_error_count":
            len(batch_errors),

        "quota_errors":
            batch_errors,
    }


# --------------------------------------------------
# Beklenmeyen batch
# --------------------------------------------------

unexpected_batches = sorted(
    set(records_by_batch)
    - set(BATCH_CONFIG)
)


# --------------------------------------------------
# Global dağılımlar
# --------------------------------------------------

category_distribution = Counter(
    r.get("category")
    for r in records
)

generator_distribution = Counter(
    r.get("generator_id")
    for r in records
)

difficulty_distribution = Counter(
    r.get("difficulty")
    for r in records
)

style_distribution = Counter(
    r.get("language_style")
    for r in records
)

location_distribution = Counter(
    r.get("location_style")
    for r in records
)

typo_distribution = Counter(
    r.get("has_typo")
    for r in records
)


total_quota_errors = sum(
    result["quota_error_count"]
    for result in batch_results.values()
)


summary = {
    "total_records":
        len(records),

    "json_error_count":
        len(json_errors),

    "record_level_error_count":
        len(errors),

    "duplicate_sample_id_count":
        len(duplicate_ids),

    "unexpected_batches":
        unexpected_batches,

    "batch_quota_error_count":
        total_quota_errors,

    "category_distribution":
        dict(category_distribution),

    "generator_distribution":
        dict(generator_distribution),

    "difficulty_distribution":
        dict(difficulty_distribution),

    "language_style_distribution":
        dict(style_distribution),

    "location_style_distribution":
        dict(location_distribution),

    "has_typo_distribution": {
        str(k): v
        for k, v
        in typo_distribution.items()
    },

    "batch_results":
        batch_results,

    "status":
        (
            "PASS"
            if (
                not json_errors
                and not errors
                and not unexpected_batches
                and total_quota_errors == 0
            )
            else "REVIEW_REQUIRED"
        ),
}


SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


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


with ERROR_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for error in errors:

        f.write(
            json.dumps(
                error,
                ensure_ascii=False,
            )
            + "\n"
        )

    for error in json_errors:

        f.write(
            json.dumps(
                {
                    "error_type":
                        "JSON_ERROR",
                    **error,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# --------------------------------------------------
# Terminal summary
# --------------------------------------------------

print("\n" + "=" * 78)
print("STRUCTURAL QA SONUCU")
print("=" * 78)

print(
    f"Toplam kayıt             : {len(records)}"
)

print(
    f"JSON hatası              : {len(json_errors)}"
)

print(
    f"Record-level hata        : {len(errors)}"
)

print(
    f"Duplicate sample_id      : {len(duplicate_ids)}"
)

print(
    f"Beklenmeyen batch        : {len(unexpected_batches)}"
)

print(
    f"Batch quota hatası       : {total_quota_errors}"
)


print("\nBatch sonuçları:")

for batch, result in sorted(
    batch_results.items()
):

    status = (
        "OK"
        if result[
            "quota_error_count"
        ] == 0
        else "REVIEW"
    )

    print(
        f"  {batch:<22} "
        f"{result['record_count']:<4} "
        f"{status}"
    )


print("\nGlobal kategori dağılımı:")

for category, count in sorted(
    category_distribution.items()
):

    print(
        f"  {category:<25} {count}"
    )


print("\nFINAL STATUS:")
print(summary["status"])


print("\nRaporlar:")

print(
    f"  {SUMMARY_FILE}"
)

print(
    f"  {ERROR_FILE}"
)