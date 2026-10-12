import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ==================================================
# Inputs
# ==================================================

FINAL_FILE = Path(
    "data/processed/master_final.jsonl"
)

TRAIN_FILE = Path(
    "data/splits/provisional/train.jsonl"
)

VAL_FILE = Path(
    "data/splits/provisional/validation.jsonl"
)

TEST_FILE = Path(
    "data/splits/provisional/test.jsonl"
)

SEMANTIC_DUPLICATE_CANDIDATES = Path(
    "ml/reports/dataset_qa/"
    "06b_semantic_duplicate_candidates.csv"
)

SCENARIO_BOUNDARY_FILE = Path(
    "ml/reports/dataset_qa/"
    "06b_scenario_boundary_top100.csv"
)

CROSS_CATEGORY_BOUNDARY_FILE = Path(
    "ml/reports/dataset_qa/"
    "06b_cross_category_top100.csv"
)


# ==================================================
# Outputs
# ==================================================

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

CANDIDATE_OUTPUT = REPORT_DIR / (
    "10a_cross_split_leakage_candidates.csv"
)

READABLE_OUTPUT = REPORT_DIR / (
    "10a_cross_split_leakage_readable.txt"
)

INFO_OUTPUT = REPORT_DIR / (
    "10a_cross_split_informational_boundaries.csv"
)

SUMMARY_OUTPUT = REPORT_DIR / (
    "10a_cross_split_leakage_summary.json"
)


# ==================================================
# Frozen/provisional hashes
# ==================================================

EXPECTED_FINAL_SHA256 = (
    "c6d8a06e0fb757c3b1db11a87959f7086fd4b22da5c97268522c5fd4d77ed1c7"
)

EXPECTED_TRAIN_SHA256 = (
    "1cc1c03db3e25a8a631267d6d16b712da9ae13059935048009c57371509ca8c4"
)

EXPECTED_VAL_SHA256 = (
    "20f7f2f0ae0652079cc82f2dc5841b0890af2a2a7636154990a3b381d87648cb"
)

EXPECTED_TEST_SHA256 = (
    "df52121b604e3934891a60df92d24c40b28c8549ec020d35ea23c9c4889baefd"
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


def load_jsonl(path):

    records = []

    with path.open(
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

    return records


def load_csv(path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:

            raise ValueError(
                f"CSV header yok: {path}"
            )

        return (
            list(reader),
            reader.fieldnames,
        )


def normalize_text(text):

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.casefold()

    # Noktalama farklarını exact-normalized
    # kontrolde etkisizleştir.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = text.replace(
        "_",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def pair_key(
    sample_id_1,
    sample_id_2,
):

    return tuple(
        sorted([
            sample_id_1,
            sample_id_2,
        ])
    )


def split_pair_name(
    split_1,
    split_2,
):

    order = {
        "TRAIN": 0,
        "VALIDATION": 1,
        "TEST": 2,
    }

    pair = sorted(
        [split_1, split_2],
        key=lambda x:
            order[x],
    )

    return (
        pair[0]
        + "_VS_"
        + pair[1]
    )


def relation_type(
    record_1,
    record_2,
):

    if (
        record_1["category"]
        != record_2["category"]
    ):

        return "DIFFERENT_CATEGORY"

    if (
        record_1["scenario"]
        == record_2["scenario"]
    ):

        return (
            "SAME_CATEGORY_SAME_SCENARIO"
        )

    return (
        "SAME_CATEGORY_DIFF_SCENARIO"
    )


def first_present(
    row,
    candidates,
):

    for name in candidates:

        if (
            name in row
            and row[name] not in {
                "",
                None,
            }
        ):

            return row[name]

    return ""


# ==================================================
# Hash integrity
# ==================================================

hash_checks = {
    "FINAL":
        (
            FINAL_FILE,
            EXPECTED_FINAL_SHA256,
        ),

    "TRAIN":
        (
            TRAIN_FILE,
            EXPECTED_TRAIN_SHA256,
        ),

    "VALIDATION":
        (
            VAL_FILE,
            EXPECTED_VAL_SHA256,
        ),

    "TEST":
        (
            TEST_FILE,
            EXPECTED_TEST_SHA256,
        ),
}


actual_hashes = {}


for name, (
    path,
    expected_hash,
) in hash_checks.items():

    actual_hash = sha256_file(
        path
    )

    actual_hashes[
        name
    ] = actual_hash

    if actual_hash != expected_hash:

        raise ValueError(
            f"{name} hash değişmiş!\n"
            f"Beklenen : {expected_hash}\n"
            f"Bulunan  : {actual_hash}\n"
            "Leakage audit durduruldu."
        )


# ==================================================
# Load final and splits
# ==================================================

final_records = load_jsonl(
    FINAL_FILE
)

train_records = load_jsonl(
    TRAIN_FILE
)

val_records = load_jsonl(
    VAL_FILE
)

test_records = load_jsonl(
    TEST_FILE
)


if len(final_records) != 1667:

    raise ValueError(
        "Final dataset count != 1667"
    )


if len(train_records) != 1167:

    raise ValueError(
        "Train count != 1167"
    )


if len(val_records) != 250:

    raise ValueError(
        "Validation count != 250"
    )


if len(test_records) != 250:

    raise ValueError(
        "Test count != 250"
    )


final_map = {
    r["sample_id"]: r
    for r in final_records
}


if len(final_map) != 1667:

    raise ValueError(
        "Final duplicate sample_id."
    )


# ==================================================
# Split membership
# ==================================================

split_map = {}


for split_name, records in [
    (
        "TRAIN",
        train_records,
    ),
    (
        "VALIDATION",
        val_records,
    ),
    (
        "TEST",
        test_records,
    ),
]:

    for record in records:

        sample_id = record[
            "sample_id"
        ]

        if sample_id in split_map:

            raise ValueError(
                "Sample birden fazla "
                f"split'te: {sample_id}"
            )

        split_map[
            sample_id
        ] = split_name


if set(split_map) != set(
    final_map
):

    raise ValueError(
        "Split union final dataset ile "
        "eşleşmiyor."
    )


# ==================================================
# Direct raw + normalized exact audit
# ==================================================

raw_groups = defaultdict(
    list
)

normalized_groups = defaultdict(
    list
)


for record in final_records:

    sample_id = record[
        "sample_id"
    ]

    text = record[
        "complaint_text"
    ]

    raw_groups[
        text
    ].append(
        sample_id
    )

    normalized_groups[
        normalize_text(text)
    ].append(
        sample_id
    )


def cross_split_groups(
    groups,
):

    results = []

    for text_key, ids in (
        groups.items()
    ):

        if len(ids) < 2:
            continue

        present_splits = {
            split_map[
                sample_id
            ]
            for sample_id in ids
        }

        if len(
            present_splits
        ) < 2:
            continue

        results.append({
            "value":
                text_key,

            "sample_ids":
                sorted(ids),

            "splits":
                sorted(
                    present_splits
                ),
        })

    return results


raw_exact_cross_split = (
    cross_split_groups(
        raw_groups
    )
)

normalized_exact_cross_split = (
    cross_split_groups(
        normalized_groups
    )
)


# ==================================================
# Load Stage 6 global candidate pools
# ==================================================

duplicate_rows, duplicate_fields = (
    load_csv(
        SEMANTIC_DUPLICATE_CANDIDATES
    )
)

scenario_rows, scenario_fields = (
    load_csv(
        SCENARIO_BOUNDARY_FILE
    )
)

cross_category_rows, cross_category_fields = (
    load_csv(
        CROSS_CATEGORY_BOUNDARY_FILE
    )
)


if len(duplicate_rows) != 305:

    raise ValueError(
        "Beklenen Stage6B duplicate "
        f"candidate=305, bulunan "
        f"{len(duplicate_rows)}."
    )


if len(scenario_rows) != 100:

    raise ValueError(
        "Beklenen scenario boundary=100, "
        f"bulunan {len(scenario_rows)}."
    )


if len(cross_category_rows) != 100:

    raise ValueError(
        "Beklenen cross-category boundary=100, "
        f"bulunan {len(cross_category_rows)}."
    )


for field_set, name in [
    (
        duplicate_fields,
        "duplicate candidates",
    ),
    (
        scenario_fields,
        "scenario boundary",
    ),
    (
        cross_category_fields,
        "cross-category boundary",
    ),
]:

    if (
        "sample_id_1"
        not in field_set
        or
        "sample_id_2"
        not in field_set
    ):

        raise ValueError(
            f"{name} dosyasında "
            "sample_id_1/sample_id_2 yok."
        )


# ==================================================
# Candidate accumulator
#
# Burada yalnızca potential leakage
# review queue tutulur.
# ==================================================

candidate_map = {}


def ensure_candidate(
    sample_id_1,
    sample_id_2,
):

    key = pair_key(
        sample_id_1,
        sample_id_2,
    )

    if key in candidate_map:

        return candidate_map[
            key
        ]

    r1 = final_map[
        key[0]
    ]

    r2 = final_map[
        key[1]
    ]

    split_1 = split_map[
        key[0]
    ]

    split_2 = split_map[
        key[1]
    ]

    candidate = {
        "sample_id_1":
            key[0],

        "split_1":
            split_1,

        "category_1":
            r1["category"],

        "scenario_1":
            r1["scenario"],

        "text_1":
            r1["complaint_text"],

        "sample_id_2":
            key[1],

        "split_2":
            split_2,

        "category_2":
            r2["category"],

        "scenario_2":
            r2["scenario"],

        "text_2":
            r2["complaint_text"],

        "split_pair":
            split_pair_name(
                split_1,
                split_2,
            ),

        "relation":
            relation_type(
                r1,
                r2,
            ),

        "raw_exact":
            False,

        "normalized_exact":
            False,

        "stage6b_duplicate_candidate":
            False,

        "semantic_similarity":
            "",

        "char_similarity":
            "",

        "word_similarity":
            "",

        "stage6_priority":
            "",

        "candidate_sources":
            set(),

        "review_requirement":
            "",

        "review_note":
            "",
    }

    candidate_map[
        key
    ] = candidate

    return candidate


# ==================================================
# Add raw exact cross-split
# ==================================================

for group in (
    raw_exact_cross_split
):

    ids = group[
        "sample_ids"
    ]

    for i in range(
        len(ids)
    ):

        for j in range(
            i + 1,
            len(ids),
        ):

            id_1 = ids[i]
            id_2 = ids[j]

            if (
                split_map[id_1]
                == split_map[id_2]
            ):
                continue

            candidate = (
                ensure_candidate(
                    id_1,
                    id_2,
                )
            )

            candidate[
                "raw_exact"
            ] = True

            candidate[
                "candidate_sources"
            ].add(
                "RAW_EXACT"
            )


# ==================================================
# Add normalized exact cross-split
# ==================================================

for group in (
    normalized_exact_cross_split
):

    ids = group[
        "sample_ids"
    ]

    for i in range(
        len(ids)
    ):

        for j in range(
            i + 1,
            len(ids),
        ):

            id_1 = ids[i]
            id_2 = ids[j]

            if (
                split_map[id_1]
                == split_map[id_2]
            ):
                continue

            candidate = (
                ensure_candidate(
                    id_1,
                    id_2,
                )
            )

            candidate[
                "normalized_exact"
            ] = True

            candidate[
                "candidate_sources"
            ].add(
                "NORMALIZED_EXACT"
            )


# ==================================================
# Filter global duplicate candidate pool
# to FINAL + CROSS-SPLIT
# ==================================================

duplicate_final_survivor_edges = 0
duplicate_cross_split_edges = 0


for row in duplicate_rows:

    id_1 = row[
        "sample_id_1"
    ]

    id_2 = row[
        "sample_id_2"
    ]


    # Stage6B master=1750 üzerinden.
    # Reject edilmiş sample'ları at.
    if (
        id_1 not in final_map
        or id_2 not in final_map
    ):

        continue


    duplicate_final_survivor_edges += 1


    if (
        split_map[id_1]
        == split_map[id_2]
    ):

        continue


    duplicate_cross_split_edges += 1


    candidate = (
        ensure_candidate(
            id_1,
            id_2,
        )
    )


    candidate[
        "stage6b_duplicate_candidate"
    ] = True

    candidate[
        "candidate_sources"
    ].add(
        "STAGE6B_DUPLICATE_POOL"
    )


    semantic_similarity = (
        first_present(
            row,
            [
                "semantic_similarity",
                "similarity",
                "cosine_similarity",
                "e5_similarity",
            ],
        )
    )

    char_similarity = (
        first_present(
            row,
            [
                "char_similarity",
                "char_cosine",
                "char_score",
                "char_tfidf_similarity",
            ],
        )
    )

    word_similarity = (
        first_present(
            row,
            [
                "word_similarity",
                "word_cosine",
                "word_score",
                "word_tfidf_similarity",
            ],
        )
    )

    priority = (
        first_present(
            row,
            [
                "priority",
                "review_priority",
                "candidate_priority",
                "semantic_priority",
            ],
        )
    )


    if semantic_similarity:

        candidate[
            "semantic_similarity"
        ] = semantic_similarity


    if char_similarity:

        candidate[
            "char_similarity"
        ] = char_similarity


    if word_similarity:

        candidate[
            "word_similarity"
        ] = word_similarity


    if priority:

        candidate[
            "stage6_priority"
        ] = priority


# ==================================================
# Informational boundary queue
#
# These were already used for human QA.
# They are NOT automatically leakage blockers.
# ==================================================

informational_rows = []


def add_informational_rows(
    rows,
    source_name,
):

    for row in rows:

        id_1 = row[
            "sample_id_1"
        ]

        id_2 = row[
            "sample_id_2"
        ]


        if (
            id_1 not in final_map
            or id_2 not in final_map
        ):

            continue


        split_1 = split_map[
            id_1
        ]

        split_2 = split_map[
            id_2
        ]


        if split_1 == split_2:
            continue


        r1 = final_map[
            id_1
        ]

        r2 = final_map[
            id_2
        ]


        similarity = (
            first_present(
                row,
                [
                    "semantic_similarity",
                    "similarity",
                    "cosine_similarity",
                    "e5_similarity",
                ],
            )
        )


        informational_rows.append({
            "source":
                source_name,

            "sample_id_1":
                id_1,

            "split_1":
                split_1,

            "category_1":
                r1["category"],

            "scenario_1":
                r1["scenario"],

            "sample_id_2":
                id_2,

            "split_2":
                split_2,

            "category_2":
                r2["category"],

            "scenario_2":
                r2["scenario"],

            "split_pair":
                split_pair_name(
                    split_1,
                    split_2,
                ),

            "relation":
                relation_type(
                    r1,
                    r2,
                ),

            "semantic_similarity":
                similarity,
        })


add_informational_rows(
    scenario_rows,
    "STAGE6B_SCENARIO_BOUNDARY_TOP100",
)

add_informational_rows(
    cross_category_rows,
    "STAGE6B_CROSS_CATEGORY_TOP100",
)


# ==================================================
# Determine which candidate pairs actually block
# automatic test freeze.
#
# Exact cross-split is always blocking.
#
# Surviving Stage6B duplicate-pool pairs with
# the SAME TARGET category require one final
# split-specific review because even legitimate
# diversity may create overly easy train-test
# near-paraphrase relationships.
#
# DIFFERENT_CATEGORY pairs are hard-negative
# boundary information, not direct leakage.
# ==================================================

for candidate in (
    candidate_map.values()
):

    if (
        candidate[
            "raw_exact"
        ]
        or candidate[
            "normalized_exact"
        ]
    ):

        candidate[
            "review_requirement"
        ] = "CRITICAL_REVIEW"

        candidate[
            "review_note"
        ] = (
            "Exact/normalized-exact text "
            "appears across different splits."
        )

        continue


    if (
        candidate[
            "stage6b_duplicate_candidate"
        ]
        and candidate[
            "category_1"
        ] == candidate[
            "category_2"
        ]
    ):

        candidate[
            "review_requirement"
        ] = "LEAKAGE_REVIEW"

        candidate[
            "review_note"
        ] = (
            "Previously identified lexical/"
            "semantic near-duplicate candidate "
            "survived QA and now spans different "
            "splits. Split-specific human review "
            "is required before test freeze."
        )

        continue


    candidate[
        "review_requirement"
    ] = "INFORMATION_ONLY"

    candidate[
        "review_note"
    ] = (
        "Cross-category similarity can be a "
        "valid hard-negative boundary and is "
        "not direct same-label leakage."
    )


# ==================================================
# Flatten source sets
# ==================================================

candidate_rows = []


for candidate in (
    candidate_map.values()
):

    row = dict(
        candidate
    )

    row[
        "candidate_sources"
    ] = "|".join(
        sorted(
            row[
                "candidate_sources"
            ]
        )
    )

    candidate_rows.append(
        row
    )


review_order = {
    "CRITICAL_REVIEW": 0,
    "LEAKAGE_REVIEW": 1,
    "INFORMATION_ONLY": 2,
}


def similarity_sort_value(row):

    value = row[
        "semantic_similarity"
    ]

    try:

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return -1.0


candidate_rows.sort(
    key=lambda row: (
        review_order[
            row[
                "review_requirement"
            ]
        ],
        -similarity_sort_value(
            row
        ),
        row[
            "sample_id_1"
        ],
        row[
            "sample_id_2"
        ],
    )
)


for rank, row in enumerate(
    candidate_rows,
    start=1,
):

    row[
        "review_rank"
    ] = rank


# ==================================================
# Review queue
# ==================================================

blocking_rows = [
    row
    for row in candidate_rows
    if row[
        "review_requirement"
    ] in {
        "CRITICAL_REVIEW",
        "LEAKAGE_REVIEW",
    }
]


information_only_rows = [
    row
    for row in candidate_rows
    if row[
        "review_requirement"
    ] == "INFORMATION_ONLY"
]


# ==================================================
# CSV output
# ==================================================

candidate_fields = [
    "review_rank",

    "review_requirement",
    "candidate_sources",

    "split_pair",
    "relation",

    "sample_id_1",
    "split_1",
    "category_1",
    "scenario_1",
    "text_1",

    "sample_id_2",
    "split_2",
    "category_2",
    "scenario_2",
    "text_2",

    "raw_exact",
    "normalized_exact",

    "stage6b_duplicate_candidate",

    "semantic_similarity",
    "char_similarity",
    "word_similarity",
    "stage6_priority",

    "review_note",
]


with CANDIDATE_OUTPUT.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=candidate_fields,
    )

    writer.writeheader()

    writer.writerows(
        candidate_rows
    )


# ==================================================
# Human-readable blocking queue
# ==================================================

with READABLE_OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    if not blocking_rows:

        f.write(
            "NO BLOCKING CROSS-SPLIT "
            "LEAKAGE CANDIDATES\n"
        )

    for index, row in enumerate(
        blocking_rows,
        start=1,
    ):

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"LEAKAGE REVIEW {index:03d}\n"
        )

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"Requirement    : "
            f"{row['review_requirement']}\n"
        )

        f.write(
            f"Sources        : "
            f"{row['candidate_sources']}\n"
        )

        f.write(
            f"Split pair     : "
            f"{row['split_pair']}\n"
        )

        f.write(
            f"Relation       : "
            f"{row['relation']}\n"
        )

        f.write(
            f"Semantic sim.  : "
            f"{row['semantic_similarity']}\n"
        )

        f.write(
            f"Char sim.      : "
            f"{row['char_similarity']}\n"
        )

        f.write(
            f"Word sim.      : "
            f"{row['word_similarity']}\n"
        )


        f.write(
            "\nA\n"
            + "-" * 100
            + "\n"
        )

        f.write(
            f"Sample ID      : "
            f"{row['sample_id_1']}\n"
        )

        f.write(
            f"Split          : "
            f"{row['split_1']}\n"
        )

        f.write(
            f"Category       : "
            f"{row['category_1']}\n"
        )

        f.write(
            f"Scenario       : "
            f"{row['scenario_1']}\n"
        )

        f.write(
            f"Text           : "
            f"{row['text_1']}\n"
        )


        f.write(
            "\nB\n"
            + "-" * 100
            + "\n"
        )

        f.write(
            f"Sample ID      : "
            f"{row['sample_id_2']}\n"
        )

        f.write(
            f"Split          : "
            f"{row['split_2']}\n"
        )

        f.write(
            f"Category       : "
            f"{row['category_2']}\n"
        )

        f.write(
            f"Scenario       : "
            f"{row['scenario_2']}\n"
        )

        f.write(
            f"Text           : "
            f"{row['text_2']}\n"
        )

        f.write(
            "\n\n"
        )


# ==================================================
# Informational boundaries CSV
# ==================================================

info_fields = [
    "source",

    "sample_id_1",
    "split_1",
    "category_1",
    "scenario_1",

    "sample_id_2",
    "split_2",
    "category_2",
    "scenario_2",

    "split_pair",
    "relation",
    "semantic_similarity",
]


with INFO_OUTPUT.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=info_fields,
    )

    writer.writeheader()

    writer.writerows(
        informational_rows
    )


# ==================================================
# Statistics
# ==================================================

review_requirement_counts = Counter(
    row[
        "review_requirement"
    ]
    for row in candidate_rows
)

relation_counts = Counter(
    row[
        "relation"
    ]
    for row in candidate_rows
)

split_pair_counts = Counter(
    row[
        "split_pair"
    ]
    for row in candidate_rows
)

blocking_split_pair_counts = Counter(
    row[
        "split_pair"
    ]
    for row in blocking_rows
)


scenario_boundary_cross_split = sum(
    1
    for row in informational_rows
    if row[
        "source"
    ]
    == "STAGE6B_SCENARIO_BOUNDARY_TOP100"
)

cross_category_boundary_cross_split = sum(
    1
    for row in informational_rows
    if row[
        "source"
    ]
    == "STAGE6B_CROSS_CATEGORY_TOP100"
)


# ==================================================
# Status
# ==================================================

critical_count = (
    review_requirement_counts.get(
        "CRITICAL_REVIEW",
        0,
    )
)

leakage_review_count = (
    review_requirement_counts.get(
        "LEAKAGE_REVIEW",
        0,
    )
)


if critical_count > 0:

    status = (
        "CRITICAL_CROSS_SPLIT_"
        "LEAKAGE_REVIEW_REQUIRED"
    )

elif leakage_review_count > 0:

    status = (
        "MANUAL_CROSS_SPLIT_"
        "LEAKAGE_REVIEW_REQUIRED"
    )

else:

    status = (
        "CROSS_SPLIT_LEAKAGE_"
        "SCREEN_PASS"
    )


# ==================================================
# Summary
# ==================================================

summary = {
    "status":
        status,

    "final_dataset_count":
        len(final_records),

    "train_count":
        len(train_records),

    "validation_count":
        len(val_records),

    "test_count":
        len(test_records),

    "total_cross_split_pair_space":
        (
            len(train_records)
            * len(val_records)
            +
            len(train_records)
            * len(test_records)
            +
            len(val_records)
            * len(test_records)
        ),

    "raw_exact_cross_split_group_count":
        len(
            raw_exact_cross_split
        ),

    "normalized_exact_cross_split_group_count":
        len(
            normalized_exact_cross_split
        ),

    "stage6b_global_duplicate_candidate_count":
        len(
            duplicate_rows
        ),

    "stage6b_duplicate_candidates_with_both_final_survivors":
        duplicate_final_survivor_edges,

    "stage6b_duplicate_candidates_cross_split":
        duplicate_cross_split_edges,

    "combined_cross_split_candidate_pair_count":
        len(
            candidate_rows
        ),

    "critical_review_pair_count":
        critical_count,

    "same_label_leakage_review_pair_count":
        leakage_review_count,

    "information_only_candidate_pair_count":
        review_requirement_counts.get(
            "INFORMATION_ONLY",
            0,
        ),

    "blocking_review_pair_count":
        len(
            blocking_rows
        ),

    "candidate_relation_distribution":
        dict(
            relation_counts
        ),

    "candidate_split_pair_distribution":
        dict(
            split_pair_counts
        ),

    "blocking_split_pair_distribution":
        dict(
            blocking_split_pair_counts
        ),

    "scenario_boundary_top100_cross_split_count":
        scenario_boundary_cross_split,

    "cross_category_top100_cross_split_count":
        cross_category_boundary_cross_split,

    "hashes_verified": {
        "final":
            actual_hashes[
                "FINAL"
            ],

        "train":
            actual_hashes[
                "TRAIN"
            ],

        "validation":
            actual_hashes[
                "VALIDATION"
            ],

        "test":
            actual_hashes[
                "TEST"
            ],
    },

    "test_frozen":
        False,

    "dataset_modified":
        False,

    "split_modified":
        False,

    "method": (
        "Cross-split leakage auditing reuses "
        "the global lexical/semantic candidate "
        "graphs generated before splitting. "
        "Pair similarity is invariant to split "
        "membership, so recalculating all pairwise "
        "TF-IDF/E5 similarities is unnecessary. "
        "Direct raw and normalized exact duplicate "
        "checks are additionally performed on the "
        "provisional split files."
    ),

    "decision_policy": (
        "Exact or normalized-exact cross-split "
        "matches are critical. Surviving Stage6B "
        "duplicate-pool pairs carrying the same "
        "target category require split-specific "
        "human review before the test set can be "
        "frozen. Different-category high-similarity "
        "pairs are treated as informational hard "
        "negative boundaries rather than direct "
        "leakage."
    ),
}


with SUMMARY_OUTPUT.open(
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
print("STAGE 10A - CROSS-SPLIT LEAKAGE AUDIT")
print("=" * 78)

print(
    f"Final dataset              : "
    f"{len(final_records)}"
)

print(
    f"Cross-split pair space     : "
    f"{summary['total_cross_split_pair_space']}"
)


print("\nHash integrity:")
print("  FINAL                    : PASS")
print("  TRAIN                    : PASS")
print("  VALIDATION               : PASS")
print("  TEST                     : PASS")


print("\nDirect exact audit:")

print(
    f"  Raw exact groups         : "
    f"{len(raw_exact_cross_split)}"
)

print(
    f"  Normalized exact groups  : "
    f"{len(normalized_exact_cross_split)}"
)


print("\nStage6B candidate graph:")

print(
    f"  Global candidates        : "
    f"{len(duplicate_rows)}"
)

print(
    f"  Both final survivors     : "
    f"{duplicate_final_survivor_edges}"
)

print(
    f"  Cross-split edges        : "
    f"{duplicate_cross_split_edges}"
)


print("\nCross-split review queue:")

print(
    f"  CRITICAL_REVIEW          : "
    f"{critical_count}"
)

print(
    f"  LEAKAGE_REVIEW           : "
    f"{leakage_review_count}"
)

print(
    f"  INFORMATION_ONLY         : "
    f"{review_requirement_counts.get('INFORMATION_ONLY', 0)}"
)

print(
    f"  BLOCKING TOTAL           : "
    f"{len(blocking_rows)}"
)


print("\nPreviously reviewed boundary information:")

print(
    f"  Scenario-boundary cross : "
    f"{scenario_boundary_cross_split}"
)

print(
    f"  Cross-category cross     : "
    f"{cross_category_boundary_cross_split}"
)


print("\nBlocking pairs by split:")

for key, value in sorted(
    blocking_split_pair_counts.items()
):

    print(
        f"  {key:<28}: "
        f"{value}"
    )


print("\nOutputs:")

print(
    f"  Candidates : "
    f"{CANDIDATE_OUTPUT}"
)

print(
    f"  Readable   : "
    f"{READABLE_OUTPUT}"
)

print(
    f"  Boundaries : "
    f"{INFO_OUTPUT}"
)

print(
    f"  Summary    : "
    f"{SUMMARY_OUTPUT}"
)


print(
    "\nTEST FROZEN: False"
)

print(
    f"STATUS: {status}"
)