import csv
import json
from collections import Counter
from pathlib import Path


SURVIVOR_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d2b_survivor_after_scenario_fix.jsonl"
)

BOUNDARY_FILE = Path(
    "ml/reports/dataset_qa/"
    "06b_cross_category_top100.csv"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

REVIEW_CSV = REPORT_DIR / (
    "07d3a_cross_category_review.csv"
)

READABLE_FILE = REPORT_DIR / (
    "07d3a_cross_category_readable.txt"
)

SUMMARY_FILE = REPORT_DIR / (
    "07d3a_cross_category_summary.json"
)


# ==================================================
# Current survivor
# ==================================================

records = []

with SURVIVOR_FILE.open(
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


if len(records) != 1681:

    raise ValueError(
        f"Beklenen survivor 1681, "
        f"bulunan {len(records)}."
    )


record_map = {
    record["sample_id"]:
        record
    for record in records
}


if len(record_map) != 1681:

    raise ValueError(
        "Survivor içinde duplicate sample_id var."
    )


survivor_ids = set(
    record_map.keys()
)


# ==================================================
# Stage 6B cross-category queue
# ==================================================

boundary_rows = []


with BOUNDARY_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    fieldnames = reader.fieldnames

    if not fieldnames:

        raise ValueError(
            "Cross-category CSV header bulunamadı."
        )


    required = {
        "sample_id_1",
        "sample_id_2",
    }


    missing = (
        required
        - set(fieldnames)
    )


    if missing:

        raise ValueError(
            "Eksik gerekli kolonlar: "
            f"{sorted(missing)}"
        )


    for row in reader:

        boundary_rows.append(
            row
        )


if len(boundary_rows) != 100:

    raise ValueError(
        f"Beklenen cross-category queue 100, "
        f"bulunan {len(boundary_rows)}."
    )


# ==================================================
# Detect similarity column
# ==================================================

similarity_candidates = [
    "semantic_similarity",
    "similarity",
    "cosine_similarity",
    "score",
]


similarity_column = None


for candidate in similarity_candidates:

    if candidate in fieldnames:

        similarity_column = candidate
        break


if similarity_column is None:

    raise ValueError(
        "Similarity kolonu bulunamadı. "
        f"Mevcut kolonlar: {fieldnames}"
    )


# ==================================================
# Filter against current survivor
# ==================================================

active_rows = []

removed_rows = []


for original_rank, row in enumerate(
    boundary_rows,
    start=1,
):

    id_1 = row[
        "sample_id_1"
    ]

    id_2 = row[
        "sample_id_2"
    ]


    # Stage 6B original master üzerinden
    # üretildiği için reject edilmiş kayıtlar
    # hâlâ burada bulunabilir.
    if (
        id_1 not in survivor_ids
        or id_2 not in survivor_ids
    ):

        removed_rows.append({
            "original_rank":
                original_rank,

            "sample_id_1":
                id_1,

            "sample_id_2":
                id_2,

            "reason":
                "PREVIOUS_STAGE_REJECT",
        })

        continue


    r1 = record_map[
        id_1
    ]

    r2 = record_map[
        id_2
    ]


    # Bu queue gerçekten cross-category olmalı.
    if (
        r1["category"]
        == r2["category"]
    ):

        raise ValueError(
            "Cross-category queue içinde "
            "aynı kategori pair bulundu:\n"
            f"{id_1} <-> {id_2}\n"
            f"{r1['category']}"
        )


    similarity = float(
        row[
            similarity_column
        ]
    )


    active_rows.append({
        "original_rank":
            original_rank,

        "review_rank":
            0,

        "semantic_similarity":
            similarity,

        "sample_id_1":
            id_1,

        "category_1":
            r1["category"],

        "scenario_1":
            r1["scenario"],

        "difficulty_1":
            r1["difficulty"],

        "style_1":
            r1["language_style"],

        "typo_1":
            r1["has_typo"],

        "location_style_1":
            r1["location_style"],

        "generator_1":
            r1["generator_id"],

        "batch_1":
            r1["generation_batch"],

        "text_1":
            r1["complaint_text"],

        "sample_id_2":
            id_2,

        "category_2":
            r2["category"],

        "scenario_2":
            r2["scenario"],

        "difficulty_2":
            r2["difficulty"],

        "style_2":
            r2["language_style"],

        "typo_2":
            r2["has_typo"],

        "location_style_2":
            r2["location_style"],

        "generator_2":
            r2["generator_id"],

        "batch_2":
            r2["generation_batch"],

        "text_2":
            r2["complaint_text"],

        # Human review fields.
        "human_decision":
            "",

        "reason_code":
            "",

        "review_note":
            "",
    })


# Highest similarity first.
active_rows.sort(
    key=lambda x:
        x["semantic_similarity"],
    reverse=True,
)


for review_rank, row in enumerate(
    active_rows,
    start=1,
):

    row[
        "review_rank"
    ] = review_rank


# ==================================================
# CSV
# ==================================================

fields = [
    "original_rank",
    "review_rank",
    "semantic_similarity",

    "sample_id_1",
    "category_1",
    "scenario_1",
    "difficulty_1",
    "style_1",
    "typo_1",
    "location_style_1",
    "generator_1",
    "batch_1",
    "text_1",

    "sample_id_2",
    "category_2",
    "scenario_2",
    "difficulty_2",
    "style_2",
    "typo_2",
    "location_style_2",
    "generator_2",
    "batch_2",
    "text_2",

    "human_decision",
    "reason_code",
    "review_note",
]


with REVIEW_CSV.open(
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
        active_rows
    )


# ==================================================
# Readable report
# ==================================================

with READABLE_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for row in active_rows:

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"CROSS CATEGORY "
            f"{row['review_rank']:03d}\n"
        )

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"Original rank : "
            f"{row['original_rank']}\n"
        )

        f.write(
            f"Similarity    : "
            f"{row['semantic_similarity']:.6f}\n"
        )


        f.write(
            "\nA\n"
            + "-" * 100
            + "\n"
        )

        f.write(
            f"Sample ID     : "
            f"{row['sample_id_1']}\n"
        )

        f.write(
            f"Category      : "
            f"{row['category_1']}\n"
        )

        f.write(
            f"Scenario      : "
            f"{row['scenario_1']}\n"
        )

        f.write(
            f"Difficulty    : "
            f"{row['difficulty_1']}\n"
        )

        f.write(
            f"Style         : "
            f"{row['style_1']}\n"
        )

        f.write(
            f"Typo          : "
            f"{row['typo_1']}\n"
        )

        f.write(
            f"Generator     : "
            f"{row['generator_1']}\n"
        )

        f.write(
            f"Text          : "
            f"{row['text_1']}\n"
        )


        f.write(
            "\nB\n"
            + "-" * 100
            + "\n"
        )

        f.write(
            f"Sample ID     : "
            f"{row['sample_id_2']}\n"
        )

        f.write(
            f"Category      : "
            f"{row['category_2']}\n"
        )

        f.write(
            f"Scenario      : "
            f"{row['scenario_2']}\n"
        )

        f.write(
            f"Difficulty    : "
            f"{row['difficulty_2']}\n"
        )

        f.write(
            f"Style         : "
            f"{row['style_2']}\n"
        )

        f.write(
            f"Typo          : "
            f"{row['typo_2']}\n"
        )

        f.write(
            f"Generator     : "
            f"{row['generator_2']}\n"
        )

        f.write(
            f"Text          : "
            f"{row['text_2']}\n"
        )

        f.write(
            "\n\n"
        )


# ==================================================
# Statistics
# ==================================================

category_pair_counts = Counter()

record_frequency = Counter()


for row in active_rows:

    category_pair = tuple(
        sorted([
            row["category_1"],
            row["category_2"],
        ])
    )


    category_pair_counts[
        " <-> ".join(
            category_pair
        )
    ] += 1


    record_frequency[
        row["sample_id_1"]
    ] += 1

    record_frequency[
        row["sample_id_2"]
    ] += 1


repeated_records = {
    sample_id:
        count
    for sample_id, count
    in record_frequency.items()
    if count > 1
}


difficulty_pair_counts = Counter()


for row in active_rows:

    difficulty_pair_counts[
        f"{row['difficulty_1']} <-> "
        f"{row['difficulty_2']}"
    ] += 1


# ==================================================
# Summary
# ==================================================

summary = {
    "stage6b_original_cross_category_queue":
        len(boundary_rows),

    "current_survivor_count":
        len(records),

    "pairs_removed_due_previous_reject":
        len(removed_rows),

    "active_cross_category_pairs":
        len(active_rows),

    "unique_records_in_active_queue":
        len(record_frequency),

    "records_appearing_in_multiple_pairs":
        len(repeated_records),

    "category_pair_distribution":
        dict(category_pair_counts),

    "difficulty_pair_distribution":
        dict(difficulty_pair_counts),

    "removed_pairs":
        removed_rows,

    "dataset_modified":
        False,

    "policy": (
        "Cross-category semantic similarity is a "
        "human-review signal, not a rejection rule. "
        "Review evaluates whether each record's main "
        "municipal problem and required intervention "
        "support its assigned closed-set category. "
        "Valid hard negatives are intentionally retained."
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
print("STAGE 7D-3A - CROSS-CATEGORY REVIEW PREP")
print("=" * 78)

print(
    f"Stage6B original queue     : "
    f"{len(boundary_rows)}"
)

print(
    f"Current survivor           : "
    f"{len(records)}"
)

print(
    f"Previous reject pair       : "
    f"{len(removed_rows)}"
)

print(
    f"Active cross-category pair : "
    f"{len(active_rows)}"
)

print(
    f"Active unique record       : "
    f"{len(record_frequency)}"
)

print(
    f"Birden fazla pair'de geçen : "
    f"{len(repeated_records)}"
)


print(
    "\nKategori sınırı dağılımı:"
)

for key, value in (
    category_pair_counts.most_common()
):

    print(
        f"  {key:<55}: {value}"
    )


print("\nDosyalar:")

print(
    f"  CSV      : {REVIEW_CSV}"
)

print(
    f"  Readable : {READABLE_FILE}"
)

print(
    f"  Summary  : {SUMMARY_FILE}"
)