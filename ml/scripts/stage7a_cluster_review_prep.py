import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ==================================================
# Paths
# ==================================================

MASTER_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

CLUSTER_FILE = Path(
    "ml/reports/dataset_qa/"
    "06b_semantic_duplicate_clusters.jsonl"
)

EMBEDDING_FILE = Path(
    "ml/reports/dataset_qa/"
    "06b_embeddings.npy"
)

PROFILE_FILE = Path(
    "ml/reports/dataset_qa/"
    "06a_semantic_similarity_summary.json"
)

LEXICAL_FILE = Path(
    "ml/reports/dataset_qa/"
    "05_lexical_candidate_pairs.csv"
)

RULE_FLAGS_FILE = Path(
    "ml/reports/dataset_qa/"
    "04_rule_flags.jsonl"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

SUMMARY_CSV = REPORT_DIR / (
    "07a_duplicate_cluster_summary.csv"
)

REVIEW_JSONL = REPORT_DIR / (
    "07a_duplicate_cluster_review.jsonl"
)

READABLE_FILE = REPORT_DIR / (
    "07a_duplicate_clusters_readable.txt"
)

SUMMARY_JSON = REPORT_DIR / (
    "07a_cluster_review_summary.json"
)


# ==================================================
# Helpers
# ==================================================

def normalize_text(text):

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.translate(
        str.maketrans({
            "I": "ı",
            "İ": "i",
            "’": "'",
            "`": "'",
            "´": "'",
        })
    )

    text = text.lower()

    text = re.sub(
        r"[^\wçğıöşü]+",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def pair_key(a, b):

    return tuple(
        sorted(
            [a, b]
        )
    )


def priority_rank(priority):

    ranks = {
        "P1": 1,
        "P2": 2,
        "P3": 3,
    }

    return ranks[
        priority
    ]


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

        record = json.loads(
            line
        )

        record["_normalized"] = (
            normalize_text(
                record[
                    "complaint_text"
                ]
            )
        )

        records.append(
            record
        )


if len(records) != 1750:

    raise ValueError(
        f"Beklenen 1750 kayıt, "
        f"bulunan {len(records)}"
    )


id_to_index = {
    record["sample_id"]: index
    for index, record
    in enumerate(records)
}

id_to_record = {
    record["sample_id"]: record
    for record in records
}


# ==================================================
# Embeddings
# ==================================================

embeddings = np.load(
    EMBEDDING_FILE
)


if embeddings.shape[0] != len(records):

    raise ValueError(
        "Embedding satır sayısı ile "
        "dataset kayıt sayısı uyuşmuyor."
    )


# ==================================================
# Empirical thresholds
# ==================================================

with PROFILE_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    profile = json.load(f)


stats = (
    profile[
        "relation_statistics"
    ][
        "SAME_CATEGORY_SAME_SCENARIO"
    ]
)


P99_5 = float(
    stats["p99_5"]
)

P99_9 = float(
    stats["p99_9"]
)


# ==================================================
# Lexical candidate map
# ==================================================

lexical_pairs = {}


with LEXICAL_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        key = pair_key(
            row["sample_id_1"],
            row["sample_id_2"],
        )

        lexical_pairs[key] = row


# ==================================================
# Rule flags
# ==================================================

rule_flags = defaultdict(
    list
)


with RULE_FLAGS_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        flag = json.loads(
            line
        )

        rule_flags[
            flag["sample_id"]
        ].append(
            flag["flag_type"]
        )


# ==================================================
# Cluster JSONL
# ==================================================

clusters = []

with CLUSTER_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        clusters.append(
            json.loads(line)
        )


print("=" * 78)
print("STAGE 7A - SEMANTIC CLUSTER REVIEW PREPARATION")
print("=" * 78)

print(
    f"Cluster sayısı : {len(clusters)}"
)

print(
    f"P99.5          : {P99_5:.6f}"
)

print(
    f"P99.9          : {P99_9:.6f}"
)


# ==================================================
# Cluster analysis
# ==================================================

review_clusters = []


for cluster in clusters:

    cluster_id = (
        cluster["cluster_id"]
    )

    member_ids = [
        member["sample_id"]
        for member
        in cluster["members"]
    ]


    indexes = [
        id_to_index[
            sample_id
        ]
        for sample_id
        in member_ids
    ]


    # ----------------------------------------------
    # Bütün cluster içi pair similarity'leri
    # ----------------------------------------------

    all_pairs = []


    for x in range(
        len(indexes)
    ):

        i = indexes[x]

        for y in range(
            x + 1,
            len(indexes),
        ):

            j = indexes[y]

            score = float(
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

            normalized_exact = (
                records[i][
                    "_normalized"
                ]
                ==
                records[j][
                    "_normalized"
                ]
            )

            lexical = (
                pair_key(
                    id_1,
                    id_2,
                )
                in lexical_pairs
            )


            all_pairs.append({
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

                "normalized_exact":
                    normalized_exact,

                "lexical_candidate":
                    lexical,
            })


    all_pairs.sort(
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
            for p in all_pairs
        ],
        dtype=np.float32,
    )


    if len(scores) == 0:

        raise ValueError(
            f"Cluster {cluster_id} "
            f"içinde pair bulunamadı."
        )


    raw_exact_count = sum(
        1
        for p in all_pairs
        if p["raw_exact"]
    )


    normalized_exact_count = sum(
        1
        for p in all_pairs
        if (
            p["normalized_exact"]
            and not p["raw_exact"]
        )
    )


    lexical_pair_count = sum(
        1
        for p in all_pairs
        if p["lexical_candidate"]
    )


    p995_count = sum(
        1
        for p in all_pairs
        if (
            p[
                "semantic_similarity"
            ]
            >= P99_5
        )
    )


    p999_count = sum(
        1
        for p in all_pairs
        if (
            p[
                "semantic_similarity"
            ]
            >= P99_9
        )
    )


    # ----------------------------------------------
    # Metadata diversity
    # ----------------------------------------------

    cluster_records = [
        id_to_record[
            sample_id
        ]
        for sample_id
        in member_ids
    ]


    generators = sorted(
        {
            r["generator_id"]
            for r
            in cluster_records
        }
    )


    batches = sorted(
        {
            r["generation_batch"]
            for r
            in cluster_records
        }
    )


    difficulties = Counter(
        r["difficulty"]
        for r
        in cluster_records
    )


    styles = Counter(
        r["language_style"]
        for r
        in cluster_records
    )


    typo_count = sum(
        1
        for r
        in cluster_records
        if r["has_typo"]
    )


    artificial_members = [
        r["sample_id"]
        for r
        in cluster_records
        if (
            "ARTIFICIAL_CONTRAST"
            in rule_flags.get(
                r["sample_id"],
                [],
            )
        )
    ]


    service_term_members = [
        r["sample_id"]
        for r
        in cluster_records
        if (
            "SERVICE_TERM"
            in rule_flags.get(
                r["sample_id"],
                [],
            )
        )
    ]


    # ----------------------------------------------
    # Review priority
    #
    # Bu REJECT kararı değildir.
    # Sadece inceleme sırasıdır.
    # ----------------------------------------------

    if (
        raw_exact_count > 0
        or normalized_exact_count > 0
        or len(
            artificial_members
        ) > 0
    ):

        review_priority = "P1"

    elif (
        p999_count > 0
        or len(
            member_ids
        ) >= 5
    ):

        review_priority = "P2"

    else:

        review_priority = "P3"


    # ----------------------------------------------
    # Possible chaining indicator
    # ----------------------------------------------

    chaining_risk = (
        float(scores.min())
        < P99_5
    )


    result = {
        "cluster_id":
            cluster_id,

        "review_priority":
            review_priority,

        "category":
            cluster["category"],

        "scenario":
            cluster["scenario"],

        "member_count":
            len(member_ids),

        "all_pair_count":
            len(all_pairs),

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

        "pairs_ge_p99_5":
            p995_count,

        "pairs_ge_p99_9":
            p999_count,

        "raw_exact_pairs":
            raw_exact_count,

        "normalized_only_exact_pairs":
            normalized_exact_count,

        "lexical_candidate_pairs":
            lexical_pair_count,

        "chaining_risk":
            chaining_risk,

        "generators":
            generators,

        "batches":
            batches,

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

        "artificial_contrast_members":
            artificial_members,

        "service_term_members":
            service_term_members,

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
                    r["location_style"],

                "rule_flags":
                    rule_flags.get(
                        r["sample_id"],
                        [],
                    ),

                "complaint_text":
                    r[
                        "complaint_text"
                    ],
            }
            for r
            in cluster_records
        ],

        # İlk 15 pair inceleme için yeterli.
        "top_pairs":
            all_pairs[:15],
    }


    review_clusters.append(
        result
    )


# ==================================================
# Sort
# ==================================================

review_clusters.sort(
    key=lambda x: (
        priority_rank(
            x[
                "review_priority"
            ]
        ),
        -x[
            "member_count"
        ],
        -x[
            "semantic_max"
        ],
    )
)


# ==================================================
# Summary CSV
# ==================================================

summary_fields = [
    "cluster_id",
    "review_priority",
    "category",
    "scenario",
    "member_count",
    "all_pair_count",
    "semantic_min",
    "semantic_mean",
    "semantic_median",
    "semantic_max",
    "pairs_ge_p99_5",
    "pairs_ge_p99_9",
    "raw_exact_pairs",
    "normalized_only_exact_pairs",
    "lexical_candidate_pairs",
    "chaining_risk",
    "generators",
    "batches",
    "difficulty_distribution",
    "style_distribution",
    "typo_member_count",
    "artificial_contrast_members",
    "service_term_members",
]


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with SUMMARY_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fields,
    )

    writer.writeheader()

    for cluster in review_clusters:

        row = {
            field:
                cluster[field]
            for field
            in summary_fields
        }

        row["generators"] = "|".join(
            cluster[
                "generators"
            ]
        )

        row["batches"] = "|".join(
            cluster[
                "batches"
            ]
        )

        row[
            "difficulty_distribution"
        ] = json.dumps(
            cluster[
                "difficulty_distribution"
            ],
            ensure_ascii=False,
        )

        row[
            "style_distribution"
        ] = json.dumps(
            cluster[
                "style_distribution"
            ],
            ensure_ascii=False,
        )

        row[
            "artificial_contrast_members"
        ] = "|".join(
            cluster[
                "artificial_contrast_members"
            ]
        )

        row[
            "service_term_members"
        ] = "|".join(
            cluster[
                "service_term_members"
            ]
        )

        writer.writerow(
            row
        )


# ==================================================
# Full JSONL
# ==================================================

with REVIEW_JSONL.open(
    "w",
    encoding="utf-8",
) as f:

    for cluster in review_clusters:

        f.write(
            json.dumps(
                cluster,
                ensure_ascii=False,
            )
            + "\n"
        )


# ==================================================
# Human-readable report
# ==================================================

with READABLE_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for cluster in review_clusters:

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"CLUSTER {cluster['cluster_id']}  "
            f"[{cluster['review_priority']}]\n"
        )

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            f"Category          : "
            f"{cluster['category']}\n"
        )

        f.write(
            f"Scenario          : "
            f"{cluster['scenario']}\n"
        )

        f.write(
            f"Member count      : "
            f"{cluster['member_count']}\n"
        )

        f.write(
            f"All pair count    : "
            f"{cluster['all_pair_count']}\n"
        )

        f.write(
            f"Semantic min      : "
            f"{cluster['semantic_min']}\n"
        )

        f.write(
            f"Semantic mean     : "
            f"{cluster['semantic_mean']}\n"
        )

        f.write(
            f"Semantic median   : "
            f"{cluster['semantic_median']}\n"
        )

        f.write(
            f"Semantic max      : "
            f"{cluster['semantic_max']}\n"
        )

        f.write(
            f">= P99.5 pairs    : "
            f"{cluster['pairs_ge_p99_5']}\n"
        )

        f.write(
            f">= P99.9 pairs    : "
            f"{cluster['pairs_ge_p99_9']}\n"
        )

        f.write(
            f"Raw exact pairs   : "
            f"{cluster['raw_exact_pairs']}\n"
        )

        f.write(
            f"Lexical pairs     : "
            f"{cluster['lexical_candidate_pairs']}\n"
        )

        f.write(
            f"Chaining risk     : "
            f"{cluster['chaining_risk']}\n"
        )

        f.write(
            f"Generators        : "
            f"{', '.join(cluster['generators'])}\n"
        )

        f.write(
            f"Artificial flags  : "
            f"{', '.join(cluster['artificial_contrast_members']) or '-'}\n"
        )

        f.write(
            "\nMEMBERS\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )


        for member in (
            cluster["members"]
        ):

            f.write(
                f"\n{member['sample_id']}\n"
            )

            f.write(
                f"  Generator : "
                f"{member['generator_id']}\n"
            )

            f.write(
                f"  Batch     : "
                f"{member['generation_batch']}\n"
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
                f"  Flags     : "
                f"{', '.join(member['rule_flags']) or '-'}\n"
            )

            f.write(
                f"  Text      : "
                f"{member['complaint_text']}\n"
            )


        f.write(
            "\nTOP PAIRS\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )


        for pair in (
            cluster["top_pairs"]
        ):

            f.write(
                f"{pair['semantic_similarity']:.6f}  "
                f"{pair['sample_id_1']}  <->  "
                f"{pair['sample_id_2']}"
            )

            if pair[
                "raw_exact"
            ]:

                f.write(
                    "  [RAW_EXACT]"
                )

            elif pair[
                "normalized_exact"
            ]:

                f.write(
                    "  [NORMALIZED_EXACT]"
                )

            if pair[
                "lexical_candidate"
            ]:

                f.write(
                    "  [LEXICAL]"
                )

            f.write(
                "\n"
            )


        f.write(
            "\n\n"
        )


# ==================================================
# Global summary
# ==================================================

priority_counts = Counter(
    cluster[
        "review_priority"
    ]
    for cluster
    in review_clusters
)

chaining_count = sum(
    1
    for cluster
    in review_clusters
    if cluster[
        "chaining_risk"
    ]
)

clusters_with_exact = sum(
    1
    for cluster
    in review_clusters
    if cluster[
        "raw_exact_pairs"
    ] > 0
    or cluster[
        "normalized_only_exact_pairs"
    ] > 0
)

clusters_with_artificial = sum(
    1
    for cluster
    in review_clusters
    if len(
        cluster[
            "artificial_contrast_members"
        ]
    ) > 0
)


summary = {
    "cluster_count":
        len(
            review_clusters
        ),

    "priority_counts":
        dict(
            priority_counts
        ),

    "clusters_with_chaining_risk":
        chaining_count,

    "clusters_with_exact_duplicates":
        clusters_with_exact,

    "clusters_with_artificial_contrast_members":
        clusters_with_artificial,

    "p99_5_threshold":
        P99_5,

    "p99_9_threshold":
        P99_9,

    "policy":
        (
            "Clusters are review units, not deletion units. "
            "Connected-component membership alone does not "
            "prove that all members are mutual duplicates. "
            "Full within-cluster pairwise similarity is "
            "measured to detect chaining before human review."
        ),
}


with SUMMARY_JSON.open(
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

print("\n" + "=" * 78)
print("STAGE 7A SONUÇ")
print("=" * 78)

print(
    f"Toplam cluster              : "
    f"{len(review_clusters)}"
)

print(
    f"P1 review                   : "
    f"{priority_counts.get('P1', 0)}"
)

print(
    f"P2 review                   : "
    f"{priority_counts.get('P2', 0)}"
)

print(
    f"P3 review                   : "
    f"{priority_counts.get('P3', 0)}"
)

print(
    f"Chaining riski olan cluster : "
    f"{chaining_count}"
)

print(
    f"Exact içeren cluster        : "
    f"{clusters_with_exact}"
)

print(
    f"Artificial flag içeren      : "
    f"{clusters_with_artificial}"
)


print("\nEn büyük 10 cluster:")

largest = sorted(
    review_clusters,
    key=lambda x:
        x["member_count"],
    reverse=True,
)[:10]


for cluster in largest:

    print(
        f"  Cluster "
        f"{cluster['cluster_id']:<3} "
        f"members={cluster['member_count']:<3} "
        f"mean={cluster['semantic_mean']:.4f} "
        f"min={cluster['semantic_min']:.4f} "
        f"max={cluster['semantic_max']:.4f} "
        f"chaining={cluster['chaining_risk']} "
        f"[{cluster['category']} / "
        f"{cluster['scenario']}]"
    )


print("\nDosyalar:")

print(
    f"  Summary CSV : {SUMMARY_CSV}"
)

print(
    f"  Review JSONL: {REVIEW_JSONL}"
)

print(
    f"  Readable    : {READABLE_FILE}"
)

print(
    f"  Summary JSON: {SUMMARY_JSON}"
)