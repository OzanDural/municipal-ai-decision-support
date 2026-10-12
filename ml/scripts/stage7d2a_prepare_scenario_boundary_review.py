import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


MASTER_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

SURVIVOR_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d1b_survivor_after_artificial.jsonl"
)

BOUNDARY_FILE = Path(
    "ml/reports/dataset_qa/"
    "06b_scenario_boundary_top100.csv"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

FILTERED_CSV = REPORT_DIR / (
    "07d2a_scenario_boundary_review.csv"
)

READABLE_FILE = REPORT_DIR / (
    "07d2a_scenario_boundary_readable.txt"
)

SUMMARY_FILE = REPORT_DIR / (
    "07d2a_scenario_boundary_summary.json"
)


# ==================================================
# Master
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
        f"Beklenen master 1750, "
        f"bulunan {len(records)}."
    )


record_map = {
    r["sample_id"]: r
    for r in records
}


# ==================================================
# Current survivor
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


if len(survivor_ids) != 1681:

    raise ValueError(
        f"Beklenen survivor 1681, "
        f"bulunan {len(survivor_ids)}."
    )


# ==================================================
# Load Stage 6B boundary queue
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
            "Boundary CSV header bulunamadı."
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
            "Boundary CSV gerekli kolonları "
            f"içermiyor: {sorted(missing)}"
        )


    for row in reader:

        boundary_rows.append(row)


if len(boundary_rows) != 100:

    raise ValueError(
        f"Beklenen Stage6B scenario "
        f"boundary queue 100, "
        f"bulunan {len(boundary_rows)}."
    )


# ==================================================
# Similarity column detection
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
        "Semantic similarity kolonu bulunamadı. "
        f"Mevcut kolonlar: {fieldnames}"
    )


# ==================================================
# Survivor filter + validation
# ==================================================

review_rows = []

removed_pairs = []


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


    if (
        id_1 not in record_map
        or id_2 not in record_map
    ):

        raise ValueError(
            "Boundary pair master dataset'te "
            "olmayan sample_id içeriyor: "
            f"{id_1}, {id_2}"
        )


    r1 = record_map[id_1]
    r2 = record_map[id_2]


    # Bu queue SAME CATEGORY /
    # DIFFERENT SCENARIO olmalı.
    if (
        r1["category"]
        != r2["category"]
    ):

        raise ValueError(
            "Scenario-boundary queue içinde "
            "cross-category pair bulundu: "
            f"{id_1} <-> {id_2}"
        )


    if (
        r1["scenario"]
        == r2["scenario"]
    ):

        raise ValueError(
            "Scenario-boundary queue içinde "
            "same-scenario pair bulundu: "
            f"{id_1} <-> {id_2}"
        )


    # Önceki QA'da reject edilmiş
    # kayıt içeren pair artık aktif
    # review queue'ya girmez.
    if (
        id_1 not in survivor_ids
        or id_2 not in survivor_ids
    ):

        removed_pairs.append({
            "original_rank":
                original_rank,

            "sample_id_1":
                id_1,

            "sample_id_2":
                id_2,
        })

        continue


    similarity = float(
        row[
            similarity_column
        ]
    )


    review_rows.append({
        "original_rank":
            original_rank,

        "review_rank":
            0,

        "semantic_similarity":
            similarity,

        "category":
            r1["category"],

        "sample_id_1":
            id_1,

        "scenario_1":
            r1["scenario"],

        "difficulty_1":
            r1["difficulty"],

        "style_1":
            r1[
                "language_style"
            ],

        "typo_1":
            r1["has_typo"],

        "generator_1":
            r1["generator_id"],

        "batch_1":
            r1[
                "generation_batch"
            ],

        "text_1":
            r1[
                "complaint_text"
            ],

        "sample_id_2":
            id_2,

        "scenario_2":
            r2["scenario"],

        "difficulty_2":
            r2["difficulty"],

        "style_2":
            r2[
                "language_style"
            ],

        "typo_2":
            r2["has_typo"],

        "generator_2":
            r2["generator_id"],

        "batch_2":
            r2[
                "generation_batch"
            ],

        "text_2":
            r2[
                "complaint_text"
            ],

        # Human review fields
        "human_decision":
            "",

        "reason_code":
            "",

        "review_note":
            "",
    })


# Similarity'e göre yeniden sırala.
review_rows.sort(
    key=lambda x:
        x[
            "semantic_similarity"
        ],
    reverse=True,
)


for rank, row in enumerate(
    review_rows,
    start=1,
):

    row[
        "review_rank"
    ] = rank


# ==================================================
# CSV
# ==================================================

fields = [
    "original_rank",
    "review_rank",
    "semantic_similarity",
    "category",
    "sample_id_1",
    "scenario_1",
    "difficulty_1",
    "style_1",
    "typo_1",
    "generator_1",
    "batch_1",
    "text_1",
    "sample_id_2",
    "scenario_2",
    "difficulty_2",
    "style_2",
    "typo_2",
    "generator_2",
    "batch_2",
    "text_2",
    "human_decision",
    "reason_code",
    "review_note",
]


with FILTERED_CSV.open(
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

    for row in review_rows:

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"SCENARIO BOUNDARY "
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
            f"Category      : "
            f"{row['category']}\n"
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
# Review statistics
# ==================================================

category_counts = Counter(
    row["category"]
    for row
    in review_rows
)


scenario_pair_counts = Counter()


for row in review_rows:

    pair = tuple(
        sorted([
            row["scenario_1"],
            row["scenario_2"],
        ])
    )

    scenario_pair_counts[
        " <-> ".join(pair)
    ] += 1


record_frequency = Counter()


for row in review_rows:

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


# ==================================================
# Summary
# ==================================================

summary = {
    "stage6b_original_queue_size":
        len(boundary_rows),

    "current_survivor_count":
        len(survivor_ids),

    "pairs_removed_due_previous_reject":
        len(removed_pairs),

    "active_scenario_boundary_pairs":
        len(review_rows),

    "unique_records_in_active_queue":
        len(record_frequency),

    "records_appearing_in_multiple_pairs":
        len(repeated_records),

    "category_distribution":
        dict(category_counts),

    "scenario_pair_distribution":
        dict(
            scenario_pair_counts
        ),

    "removed_pairs":
        removed_pairs,

    "dataset_modified":
        False,

    "policy": (
        "Scenario-boundary pairs contain the same "
        "classification category but different QA "
        "scenario metadata. High semantic similarity "
        "is not treated as a deletion criterion. "
        "Review focuses on scenario consistency, "
        "metadata correctness, ambiguity, and useful "
        "boundary diversity."
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
print("STAGE 7D-2A - SCENARIO BOUNDARY REVIEW PREP")
print("=" * 78)

print(
    f"Stage6B original queue       : "
    f"{len(boundary_rows)}"
)

print(
    f"Current survivor             : "
    f"{len(survivor_ids)}"
)

print(
    f"Previous reject nedeniyle "
    f"elenen pair                 : "
    f"{len(removed_pairs)}"
)

print(
    f"Active boundary pair         : "
    f"{len(review_rows)}"
)

print(
    f"Active unique record         : "
    f"{len(record_frequency)}"
)

print(
    f"Birden fazla pair'de geçen   : "
    f"{len(repeated_records)}"
)


print("\nKategori dağılımı:")

for key, value in sorted(
    category_counts.items()
):

    print(
        f"  {key:<24}: {value}"
    )


print("\nEn sık scenario boundary türleri:")

for key, value in (
    scenario_pair_counts.most_common(
        10
    )
):

    print(
        f"  {key:<55}: {value}"
    )


print("\nDosyalar:")

print(
    f"  CSV      : {FILTERED_CSV}"
)

print(
    f"  Readable : {READABLE_FILE}"
)

print(
    f"  Summary  : {SUMMARY_FILE}"
)