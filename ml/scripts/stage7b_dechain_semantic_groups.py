import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


MASTER_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

EMBEDDING_FILE = Path(
    "ml/reports/dataset_qa/06b_embeddings.npy"
)

PROFILE_FILE = Path(
    "ml/reports/dataset_qa/06a_semantic_similarity_summary.json"
)

CANDIDATE_FILE = Path(
    "ml/reports/dataset_qa/06b_semantic_duplicate_candidates.csv"
)

LEXICAL_FILE = Path(
    "ml/reports/dataset_qa/05_lexical_candidate_pairs.csv"
)

STAGE6B_SUMMARY = Path(
    "ml/reports/dataset_qa/06b_semantic_candidate_summary.json"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

GROUPS_FILE = REPORT_DIR / (
    "07b_complete_linkage_groups.jsonl"
)

GROUP_SUMMARY_CSV = REPORT_DIR / (
    "07b_complete_linkage_group_summary.csv"
)

READABLE_FILE = REPORT_DIR / (
    "07b_complete_linkage_readable.txt"
)

RESIDUAL_FILE = REPORT_DIR / (
    "07b_residual_pair_review.csv"
)

GROUP_MAP_FILE = REPORT_DIR / (
    "07b_complete_linkage_group_map.json"
)

SUMMARY_FILE = REPORT_DIR / (
    "07b_dechain_summary.json"
)


TOLERANCE = 1e-5


def pair_key(a, b):
    return tuple(sorted([a, b]))


def safe_score(value):
    return float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


# --------------------------------------------------
# Master dataset
# --------------------------------------------------

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
        f"Beklenen 1750 kayıt, "
        f"bulunan {len(records)}."
    )


id_to_index = {
    r["sample_id"]: i
    for i, r in enumerate(records)
}


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = np.load(
    EMBEDDING_FILE
).astype(
    np.float32
)


if embeddings.shape[0] != len(records):
    raise ValueError(
        "Embedding satır sayısı dataset ile uyuşmuyor."
    )


norms = np.linalg.norm(
    embeddings,
    axis=1,
)


if not np.allclose(
    norms,
    1.0,
    atol=1e-3,
):
    raise ValueError(
        "Embedding'ler normalize görünmüyor."
    )


# --------------------------------------------------
# Empirical thresholds
# --------------------------------------------------

with PROFILE_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    profile = json.load(f)


same_scenario_stats = (
    profile[
        "relation_statistics"
    ][
        "SAME_CATEGORY_SAME_SCENARIO"
    ]
)


P99_5 = float(
    same_scenario_stats["p99_5"]
)

P99_9 = float(
    same_scenario_stats["p99_9"]
)


DISTANCE_THRESHOLD = (
    1.0 - P99_5
)


# --------------------------------------------------
# Stage 5 lexical pair map
# --------------------------------------------------

lexical_pairs = {}


with LEXICAL_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        lexical_pairs[
            pair_key(
                row["sample_id_1"],
                row["sample_id_2"],
            )
        ] = row


# --------------------------------------------------
# Category + scenario grupları
# --------------------------------------------------

scenario_groups = defaultdict(
    list
)


for index, record in enumerate(
    records
):

    key = (
        record["category"],
        record["scenario"],
    )

    scenario_groups[
        key
    ].append(
        index
    )


# --------------------------------------------------
# Complete-linkage
# --------------------------------------------------

complete_groups = []

sample_to_group = {}

next_group_id = 1


for (
    category,
    scenario
), indexes in sorted(
    scenario_groups.items()
):

    if len(indexes) < 2:
        continue


    group_embeddings = (
        embeddings[indexes]
    )


    # Cosine distance:
    # 1 - cosine similarity
    condensed = pdist(
        group_embeddings,
        metric="cosine",
    )


    if len(condensed) == 0:
        continue


    hierarchy = linkage(
        condensed,
        method="complete",
    )


    labels = fcluster(
        hierarchy,
        t=(
            DISTANCE_THRESHOLD
            + 1e-12
        ),
        criterion="distance",
    )


    label_members = defaultdict(
        list
    )


    for local_index, label in enumerate(
        labels
    ):

        label_members[
            int(label)
        ].append(
            indexes[
                local_index
            ]
        )


    for members in (
        label_members.values()
    ):

        if len(members) < 2:
            continue


        pair_rows = []


        for x in range(
            len(members)
        ):

            i = members[x]

            for y in range(
                x + 1,
                len(members),
            ):

                j = members[y]

                score = safe_score(
                    np.dot(
                        embeddings[i],
                        embeddings[j],
                    )
                )

                id_1 = records[i][
                    "sample_id"
                ]

                id_2 = records[j][
                    "sample_id"
                ]

                raw_exact = (
                    records[i][
                        "complaint_text"
                    ]
                    ==
                    records[j][
                        "complaint_text"
                    ]
                )

                lexical = (
                    pair_key(
                        id_1,
                        id_2,
                    )
                    in lexical_pairs
                )


                pair_rows.append({
                    "sample_id_1":
                        id_1,

                    "sample_id_2":
                        id_2,

                    "semantic_similarity":
                        round(
                            score,
                            6,
                        ),

                    "raw_exact":
                        raw_exact,

                    "lexical_candidate":
                        lexical,
                })


        pair_rows.sort(
            key=lambda x:
                x[
                    "semantic_similarity"
                ],
            reverse=True,
        )


        scores = np.array(
            [
                p[
                    "semantic_similarity"
                ]
                for p in pair_rows
            ],
            dtype=np.float32,
        )


        semantic_min = float(
            scores.min()
        )


        # Complete-linkage'in gerçekten
        # istediğimiz koşulu sağladığını doğrula.
        if (
            semantic_min
            < P99_5 - TOLERANCE
        ):

            raise ValueError(
                f"Complete-linkage doğrulama hatası: "
                f"{category}/{scenario}, "
                f"min={semantic_min}, "
                f"threshold={P99_5}"
            )


        raw_exact_pairs = sum(
            p["raw_exact"]
            for p in pair_rows
        )

        lexical_pair_count = sum(
            p["lexical_candidate"]
            for p in pair_rows
        )


        member_records = [
            records[i]
            for i in members
        ]


        original_count = sum(
            1
            for r in member_records
            if not r[
                "generation_batch"
            ].startswith(
                "long_"
            )
        )

        long_count = (
            len(member_records)
            - original_count
        )


        generators = sorted(
            {
                r["generator_id"]
                for r
                in member_records
            }
        )

        difficulties = Counter(
            r["difficulty"]
            for r
            in member_records
        )

        styles = Counter(
            r["language_style"]
            for r
            in member_records
        )

        typo_count = sum(
            1
            for r in member_records
            if r["has_typo"]
        )


        if raw_exact_pairs > 0:

            review_priority = (
                "R1_EXACT"
            )

        elif (
            semantic_min
            >= P99_9
        ):

            review_priority = (
                "R1_VERY_HIGH_COHESION"
            )

        elif (
            len(members) >= 3
            or lexical_pair_count > 0
        ):

            review_priority = (
                "R2_COHESIVE"
            )

        else:

            review_priority = (
                "R3_PAIR"
            )


        group_id = (
            f"CLG_{next_group_id:03d}"
        )

        next_group_id += 1


        for i in members:

            sample_id = records[i][
                "sample_id"
            ]

            # Complete-linkage bir partition olduğu
            # için bir kayıt en fazla bir gruptadır.
            if sample_id in sample_to_group:
                raise ValueError(
                    f"{sample_id} birden fazla "
                    f"complete-linkage grubuna girdi."
                )

            sample_to_group[
                sample_id
            ] = group_id


        item = {
            "group_id":
                group_id,

            "review_priority":
                review_priority,

            "category":
                category,

            "scenario":
                scenario,

            "member_count":
                len(members),

            "all_pair_count":
                len(pair_rows),

            "semantic_min":
                round(
                    float(scores.min()),
                    6,
                ),

            "semantic_mean":
                round(
                    float(scores.mean()),
                    6,
                ),

            "semantic_median":
                round(
                    float(
                        np.median(scores)
                    ),
                    6,
                ),

            "semantic_max":
                round(
                    float(scores.max()),
                    6,
                ),

            "pairs_ge_p99_9":
                sum(
                    1
                    for score in scores
                    if score >= P99_9
                ),

            "raw_exact_pairs":
                raw_exact_pairs,

            "lexical_candidate_pairs":
                lexical_pair_count,

            "original_member_count":
                original_count,

            "long_member_count":
                long_count,

            "generators":
                generators,

            "difficulty_distribution":
                dict(
                    difficulties
                ),

            "style_distribution":
                dict(
                    styles
                ),

            "typo_member_count":
                typo_count,

            "members": [
                {
                    "sample_id":
                        r["sample_id"],

                    "generator_id":
                        r["generator_id"],

                    "generation_batch":
                        r[
                            "generation_batch"
                        ],

                    "difficulty":
                        r["difficulty"],

                    "language_style":
                        r[
                            "language_style"
                        ],

                    "has_typo":
                        r["has_typo"],

                    "location_style":
                        r[
                            "location_style"
                        ],

                    "complaint_text":
                        r[
                            "complaint_text"
                        ],
                }
                for r
                in member_records
            ],

            "pairwise_similarities":
                pair_rows,
        }


        complete_groups.append(
            item
        )


# --------------------------------------------------
# Sort review groups
# --------------------------------------------------

priority_order = {
    "R1_EXACT": 1,
    "R1_VERY_HIGH_COHESION": 2,
    "R2_COHESIVE": 3,
    "R3_PAIR": 4,
}


complete_groups.sort(
    key=lambda x: (
        priority_order[
            x["review_priority"]
        ],
        -x["member_count"],
        -x["semantic_min"],
    )
)


# --------------------------------------------------
# Stage 6B candidate pairs:
# complete-linkage dışında kalan hiçbir adayı
# kaybetme.
# --------------------------------------------------

candidate_rows = []


with CANDIDATE_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        candidate_rows.append(
            row
        )


covered_candidate_count = 0
residual_rows = []


for row in candidate_rows:

    id_1 = row[
        "sample_id_1"
    ]

    id_2 = row[
        "sample_id_2"
    ]

    group_1 = (
        sample_to_group.get(
            id_1
        )
    )

    group_2 = (
        sample_to_group.get(
            id_2
        )
    )


    if (
        group_1 is not None
        and group_1 == group_2
    ):

        covered_candidate_count += 1
        continue


    semantic_score = float(
        row[
            "semantic_similarity"
        ]
    )

    lexical_candidate = (
        str(
            row[
                "lexical_candidate"
            ]
        ).lower()
        == "true"
    )


    if (
        semantic_score
        >= P99_5
    ):

        residual_reason = (
            "SEMANTIC_EDGE_CROSS_COMPLETE_GROUP"
        )

    elif lexical_candidate:

        residual_reason = (
            "LEXICAL_ONLY_BELOW_P99_5"
        )

    else:

        residual_reason = (
            "OTHER_REVIEW"
        )


    residual_rows.append({
        **row,

        "complete_group_1":
            group_1 or "",

        "complete_group_2":
            group_2 or "",

        "residual_reason":
            residual_reason,
    })


# Her Stage6B candidate mutlaka ya
# complete group içinde ya residual olmalı.
if (
    covered_candidate_count
    + len(residual_rows)
    != len(candidate_rows)
):

    raise ValueError(
        "Candidate coverage kontrolü başarısız."
    )


# --------------------------------------------------
# Outputs
# --------------------------------------------------

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with GROUPS_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for item in complete_groups:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


summary_fields = [
    "group_id",
    "review_priority",
    "category",
    "scenario",
    "member_count",
    "all_pair_count",
    "semantic_min",
    "semantic_mean",
    "semantic_median",
    "semantic_max",
    "pairs_ge_p99_9",
    "raw_exact_pairs",
    "lexical_candidate_pairs",
    "original_member_count",
    "long_member_count",
    "generators",
    "difficulty_distribution",
    "style_distribution",
    "typo_member_count",
]


with GROUP_SUMMARY_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fields,
    )

    writer.writeheader()


    for group in complete_groups:

        row = {
            key:
                group[key]
            for key
            in summary_fields
        }

        row["generators"] = (
            "|".join(
                group[
                    "generators"
                ]
            )
        )

        row[
            "difficulty_distribution"
        ] = json.dumps(
            group[
                "difficulty_distribution"
            ],
            ensure_ascii=False,
        )

        row[
            "style_distribution"
        ] = json.dumps(
            group[
                "style_distribution"
            ],
            ensure_ascii=False,
        )

        writer.writerow(
            row
        )


with RESIDUAL_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    if residual_rows:

        fields = list(
            residual_rows[0].keys()
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            residual_rows
        )


with GROUP_MAP_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        sample_to_group,
        f,
        ensure_ascii=False,
        indent=2,
    )


# --------------------------------------------------
# Human-readable output
# --------------------------------------------------

with READABLE_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for group in complete_groups:

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"{group['group_id']} "
            f"[{group['review_priority']}]\n"
        )

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"Category       : "
            f"{group['category']}\n"
        )

        f.write(
            f"Scenario       : "
            f"{group['scenario']}\n"
        )

        f.write(
            f"Members        : "
            f"{group['member_count']}\n"
        )

        f.write(
            f"Semantic min   : "
            f"{group['semantic_min']}\n"
        )

        f.write(
            f"Semantic mean  : "
            f"{group['semantic_mean']}\n"
        )

        f.write(
            f"Semantic max   : "
            f"{group['semantic_max']}\n"
        )

        f.write(
            f"Exact pairs    : "
            f"{group['raw_exact_pairs']}\n"
        )

        f.write(
            f"Lexical pairs  : "
            f"{group['lexical_candidate_pairs']}\n"
        )

        f.write(
            f"Original / long: "
            f"{group['original_member_count']} / "
            f"{group['long_member_count']}\n"
        )


        f.write(
            "\nMEMBERS\n"
            + "-" * 100
            + "\n"
        )


        for member in (
            group["members"]
        ):

            f.write(
                f"\n{member['sample_id']}\n"
            )

            f.write(
                f"  Difficulty: "
                f"{member['difficulty']}\n"
            )

            f.write(
                f"  Style     : "
                f"{member['language_style']}\n"
            )

            f.write(
                f"  Typo      : "
                f"{member['has_typo']}\n"
            )

            f.write(
                f"  Text      : "
                f"{member['complaint_text']}\n"
            )


        f.write(
            "\nPAIRWISE\n"
            + "-" * 100
            + "\n"
        )


        for pair in (
            group[
                "pairwise_similarities"
            ]
        ):

            extras = []

            if pair["raw_exact"]:
                extras.append(
                    "RAW_EXACT"
                )

            if pair[
                "lexical_candidate"
            ]:
                extras.append(
                    "LEXICAL"
                )

            suffix = (
                " [" +
                ", ".join(extras)
                + "]"
                if extras
                else ""
            )


            f.write(
                f"{pair['semantic_similarity']:.6f}  "
                f"{pair['sample_id_1']}  <->  "
                f"{pair['sample_id_2']}"
                f"{suffix}\n"
            )


        f.write(
            "\n\n"
        )


# --------------------------------------------------
# Compare Stage 6B
# --------------------------------------------------

with STAGE6B_SUMMARY.open(
    "r",
    encoding="utf-8",
) as f:

    stage6b = json.load(f)


unique_complete_members = len(
    sample_to_group
)


largest_complete_group = max(
    (
        group[
            "member_count"
        ]
        for group
        in complete_groups
    ),
    default=0,
)


priority_counts = Counter(
    group[
        "review_priority"
    ]
    for group
    in complete_groups
)


residual_reason_counts = Counter(
    row[
        "residual_reason"
    ]
    for row
    in residual_rows
)


summary = {
    "p99_5_similarity_threshold":
        P99_5,

    "p99_9_similarity_threshold":
        P99_9,

    "complete_linkage_distance_threshold":
        DISTANCE_THRESHOLD,

    "stage6b_connected_component_count":
        stage6b[
            "semantic_cluster_count"
        ],

    "stage6b_records_in_components":
        stage6b[
            "records_in_semantic_clusters"
        ],

    "stage6b_largest_component":
        stage6b[
            "largest_cluster_size"
        ],

    "complete_linkage_group_count":
        len(
            complete_groups
        ),

    "records_in_complete_linkage_groups":
        unique_complete_members,

    "largest_complete_linkage_group":
        largest_complete_group,

    "review_priority_counts":
        dict(
            priority_counts
        ),

    "stage6b_candidate_pair_count":
        len(
            candidate_rows
        ),

    "candidate_pairs_covered_by_complete_groups":
        covered_candidate_count,

    "residual_pair_count":
        len(
            residual_rows
        ),

    "residual_reason_counts":
        dict(
            residual_reason_counts
        ),

    "policy":
        (
            "Complete-linkage groups are cohesive human-review "
            "units, not automatic deletion groups. Every member "
            "within a complete-linkage group satisfies the "
            "empirical P99.5 pairwise similarity requirement. "
            "Stage 6B candidate pairs not represented inside "
            "one complete-linkage group are retained in the "
            "residual review queue, so no candidate evidence "
            "is discarded."
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


# --------------------------------------------------
# Terminal
# --------------------------------------------------

print("=" * 78)
print("STAGE 7B - DE-CHAIN WITH COMPLETE LINKAGE")
print("=" * 78)

print(
    f"P99.5 similarity        : "
    f"{P99_5:.6f}"
)

print(
    f"P99.9 similarity        : "
    f"{P99_9:.6f}"
)

print("\nStage 6B connected components:")

print(
    f"  Component             : "
    f"{stage6b['semantic_cluster_count']}"
)

print(
    f"  Records               : "
    f"{stage6b['records_in_semantic_clusters']}"
)

print(
    f"  Largest               : "
    f"{stage6b['largest_cluster_size']}"
)


print("\nComplete-linkage review groups:")

print(
    f"  Group                 : "
    f"{len(complete_groups)}"
)

print(
    f"  Unique records        : "
    f"{unique_complete_members}"
)

print(
    f"  Largest               : "
    f"{largest_complete_group}"
)


print("\nReview priority:")

for key, value in sorted(
    priority_counts.items()
):

    print(
        f"  {key:<24}: "
        f"{value}"
    )


print("\nCandidate coverage:")

print(
    f"  Stage6B candidates    : "
    f"{len(candidate_rows)}"
)

print(
    f"  Covered in groups     : "
    f"{covered_candidate_count}"
)

print(
    f"  Residual pair review  : "
    f"{len(residual_rows)}"
)


print("\nResidual reasons:")

for key, value in sorted(
    residual_reason_counts.items()
):

    print(
        f"  {key:<40}: "
        f"{value}"
    )


print("\nEn büyük 10 complete-linkage group:")

for group in sorted(
    complete_groups,
    key=lambda x:
        x["member_count"],
    reverse=True,
)[:10]:

    print(
        f"  {group['group_id']}  "
        f"members={group['member_count']:<3} "
        f"min={group['semantic_min']:.4f} "
        f"mean={group['semantic_mean']:.4f} "
        f"max={group['semantic_max']:.4f} "
        f"[{group['category']} / "
        f"{group['scenario']}]"
    )


print("\nDosyalar:")

print(
    f"  Groups   : {GROUPS_FILE}"
)

print(
    f"  Summary  : {GROUP_SUMMARY_CSV}"
)

print(
    f"  Readable : {READABLE_FILE}"
)

print(
    f"  Residual : {RESIDUAL_FILE}"
)

print(
    f"  Group map: {GROUP_MAP_FILE}"
)

print(
    f"  Audit    : {SUMMARY_FILE}"
)