import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


MASTER_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

SURVIVOR_FILE = Path(
    "ml/reports/dataset_qa/"
    "07c_duplicate_review_survivor.jsonl"
)

RULE_FLAGS_FILE = Path(
    "ml/reports/dataset_qa/"
    "04_rule_flags.jsonl"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

CSV_FILE = REPORT_DIR / (
    "07d1_artificial_contrast_review.csv"
)

READABLE_FILE = REPORT_DIR / (
    "07d1_artificial_contrast_readable.txt"
)

SUMMARY_FILE = REPORT_DIR / (
    "07d1_artificial_contrast_summary.json"
)


# ==================================================
# Master dataset
# ==================================================

records = []

with MASTER_FILE.open(
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


if len(records) != 1750:

    raise ValueError(
        f"Beklenen 1750 master kayıt, "
        f"bulunan {len(records)}."
    )


record_map = {
    record["sample_id"]:
        record
    for record in records
}


# ==================================================
# Duplicate-QA survivor
# ==================================================

survivor_ids = set()


with SURVIVOR_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        survivor_ids.add(
            record["sample_id"]
        )


if len(survivor_ids) != 1692:

    raise ValueError(
        f"Beklenen survivor 1692, "
        f"bulunan {len(survivor_ids)}."
    )


# ==================================================
# Rule flags
# ==================================================

all_flags_by_sample = defaultdict(
    list
)

artificial_flags_by_sample = defaultdict(
    list
)

artificial_flag_occurrences = 0


with RULE_FLAGS_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        flag = json.loads(line)

        sample_id = flag[
            "sample_id"
        ]

        flag_type = flag[
            "flag_type"
        ]


        all_flags_by_sample[
            sample_id
        ].append(
            flag
        )


        if (
            flag_type
            == "ARTIFICIAL_CONTRAST"
        ):

            artificial_flag_occurrences += 1

            artificial_flags_by_sample[
                sample_id
            ].append(
                flag
            )


# ==================================================
# Unique artificial records
# ==================================================

all_artificial_ids = set(
    artificial_flags_by_sample.keys()
)


surviving_artificial_ids = (
    all_artificial_ids
    & survivor_ids
)


removed_by_duplicate_ids = (
    all_artificial_ids
    - survivor_ids
)


# ==================================================
# Review records
# ==================================================

review_rows = []


for sample_id in sorted(
    surviving_artificial_ids
):

    if sample_id not in record_map:

        raise ValueError(
            f"Rule flag sample_id "
            f"master dataset'te yok: "
            f"{sample_id}"
        )


    record = record_map[
        sample_id
    ]


    all_flag_types = sorted(
        {
            flag["flag_type"]
            for flag
            in all_flags_by_sample[
                sample_id
            ]
        }
    )


    artificial_raw = (
        artificial_flags_by_sample[
            sample_id
        ]
    )


    review_rows.append({
        "sample_id":
            sample_id,

        "category":
            record["category"],

        "scenario":
            record["scenario"],

        "difficulty":
            record["difficulty"],

        "language_style":
            record[
                "language_style"
            ],

        "has_typo":
            record["has_typo"],

        "location_style":
            record[
                "location_style"
            ],

        "generator_id":
            record[
                "generator_id"
            ],

        "generation_batch":
            record[
                "generation_batch"
            ],

        "all_flag_types":
            "|".join(
                all_flag_types
            ),

        "artificial_flag_count":
            len(
                artificial_raw
            ),

        "artificial_flag_details":
            json.dumps(
                artificial_raw,
                ensure_ascii=False,
            ),

        "complaint_text":
            record[
                "complaint_text"
            ],

        # Bunları sonraki insan
        # incelemesinde dolduracağız.
        "human_decision":
            "",

        "reason_code":
            "",

        "review_note":
            "",
    })


# ==================================================
# CSV
# ==================================================

fields = [
    "sample_id",
    "category",
    "scenario",
    "difficulty",
    "language_style",
    "has_typo",
    "location_style",
    "generator_id",
    "generation_batch",
    "all_flag_types",
    "artificial_flag_count",
    "artificial_flag_details",
    "complaint_text",
    "human_decision",
    "reason_code",
    "review_note",
]


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with CSV_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    writer.writerows(
        review_rows
    )


# ==================================================
# Readable report
# ==================================================

with READABLE_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for index, row in enumerate(
        review_rows,
        start=1,
    ):

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"ARTIFICIAL REVIEW "
            f"{index:02d}\n"
        )

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"Sample ID       : "
            f"{row['sample_id']}\n"
        )

        f.write(
            f"Category        : "
            f"{row['category']}\n"
        )

        f.write(
            f"Scenario        : "
            f"{row['scenario']}\n"
        )

        f.write(
            f"Difficulty      : "
            f"{row['difficulty']}\n"
        )

        f.write(
            f"Style           : "
            f"{row['language_style']}\n"
        )

        f.write(
            f"Typo            : "
            f"{row['has_typo']}\n"
        )

        f.write(
            f"Generator       : "
            f"{row['generator_id']}\n"
        )

        f.write(
            f"Batch           : "
            f"{row['generation_batch']}\n"
        )

        f.write(
            f"All flags       : "
            f"{row['all_flag_types']}\n"
        )

        f.write(
            f"Artificial hits : "
            f"{row['artificial_flag_count']}\n"
        )

        f.write(
            "\nTEXT\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        f.write(
            row[
                "complaint_text"
            ]
            + "\n"
        )


        f.write(
            "\nFLAG DETAILS\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )


        for flag in json.loads(
            row[
                "artificial_flag_details"
            ]
        ):

            f.write(
                json.dumps(
                    flag,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )


        f.write(
            "\n\n"
        )


# ==================================================
# Summary
# ==================================================

category_counts = Counter(
    row["category"]
    for row
    in review_rows
)

difficulty_counts = Counter(
    row["difficulty"]
    for row
    in review_rows
)


summary = {
    "master_record_count":
        len(records),

    "duplicate_review_survivor_count":
        len(survivor_ids),

    "artificial_flag_occurrence_count":
        artificial_flag_occurrences,

    "unique_artificial_record_count_before_duplicate_filter":
        len(
            all_artificial_ids
        ),

    "artificial_records_removed_as_duplicates":
        len(
            removed_by_duplicate_ids
        ),

    "unique_artificial_record_count_for_review":
        len(
            surviving_artificial_ids
        ),

    "category_distribution":
        dict(
            category_counts
        ),

    "difficulty_distribution":
        dict(
            difficulty_counts
        ),

    "removed_by_duplicate_sample_ids":
        sorted(
            removed_by_duplicate_ids
        ),

    "dataset_modified":
        False,

    "policy": (
        "Rule-based artificial-contrast flags "
        "are review signals only. Records are "
        "not automatically rejected. Human review "
        "must determine whether wording unnaturally "
        "reveals or prioritizes the target class."
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
print("STAGE 7D-1 - ARTIFICIAL CONTRAST REVIEW PREP")
print("=" * 78)

print(
    f"Master kayıt                      : "
    f"{len(records)}"
)

print(
    f"Duplicate survivor                : "
    f"{len(survivor_ids)}"
)

print(
    f"ARTIFICIAL_CONTRAST flag          : "
    f"{artificial_flag_occurrences}"
)

print(
    f"Unique artificial record          : "
    f"{len(all_artificial_ids)}"
)

print(
    f"Duplicate review'da elenen        : "
    f"{len(removed_by_duplicate_ids)}"
)

print(
    f"İncelenecek artificial record     : "
    f"{len(surviving_artificial_ids)}"
)


print("\nKategori dağılımı:")

for key, value in sorted(
    category_counts.items()
):

    print(
        f"  {key:<24}: {value}"
    )


if removed_by_duplicate_ids:

    print(
        "\nDuplicate nedeniyle "
        "artık incelenmeyecek:"
    )

    for sample_id in sorted(
        removed_by_duplicate_ids
    ):

        print(
            f"  {sample_id}"
        )


print("\nDosyalar:")

print(
    f"  CSV      : {CSV_FILE}"
)

print(
    f"  Readable : {READABLE_FILE}"
)

print(
    f"  Summary  : {SUMMARY_FILE}"
)