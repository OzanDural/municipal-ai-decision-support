import csv
import hashlib
import heapq
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

PROFILE_FILE = Path(
    "ml/reports/dataset_qa/06a_semantic_similarity_summary.json"
)

LEXICAL_FILE = Path(
    "ml/reports/dataset_qa/05_lexical_candidate_pairs.csv"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

DUPLICATE_FILE = REPORT_DIR / (
    "06b_semantic_duplicate_candidates.csv"
)

CLUSTER_FILE = REPORT_DIR / (
    "06b_semantic_duplicate_clusters.jsonl"
)

SCENARIO_BOUNDARY_FILE = REPORT_DIR / (
    "06b_scenario_boundary_top100.csv"
)

CATEGORY_BOUNDARY_FILE = REPORT_DIR / (
    "06b_cross_category_top100.csv"
)

SUMMARY_FILE = REPORT_DIR / (
    "06b_semantic_candidate_summary.json"
)

EMBEDDING_FILE = REPORT_DIR / (
    "06b_embeddings.npy"
)


MODEL_NAME = (
    "intfloat/multilingual-e5-base"
)

SCENARIO_BOUNDARY_TOP_K = 100
CATEGORY_BOUNDARY_TOP_K = 100


# --------------------------------------------------
# Helpers
# --------------------------------------------------

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


def push_top_k(
    heap,
    item,
    k,
):

    score = item[0]

    if len(heap) < k:

        heapq.heappush(
            heap,
            item,
        )

    elif score > heap[0][0]:

        heapq.heapreplace(
            heap,
            item,
        )


# --------------------------------------------------
# Union-Find
# --------------------------------------------------

class UnionFind:

    def __init__(self, size):

        self.parent = list(
            range(size)
        )

        self.rank = [
            0
            for _ in range(size)
        ]

    def find(self, x):

        if self.parent[x] != x:

            self.parent[x] = (
                self.find(
                    self.parent[x]
                )
            )

        return self.parent[x]

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if (
            self.rank[root_a]
            < self.rank[root_b]
        ):

            root_a, root_b = (
                root_b, root_a
            )

        self.parent[root_b] = (
            root_a
        )

        if (
            self.rank[root_a]
            == self.rank[root_b]
        ):

            self.rank[root_a] += 1


# --------------------------------------------------
# Dataset
# --------------------------------------------------

records = []

with INPUT_FILE.open(
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


# --------------------------------------------------
# Stage 6A empirical thresholds
# --------------------------------------------------

with PROFILE_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    profile = json.load(
        f
    )


same_scenario_stats = (
    profile[
        "relation_statistics"
    ][
        "SAME_CATEGORY_SAME_SCENARIO"
    ]
)


DUPLICATE_THRESHOLD = float(
    same_scenario_stats[
        "p99_5"
    ]
)

HIGH_PRIORITY_THRESHOLD = float(
    same_scenario_stats[
        "p99_9"
    ]
)


# --------------------------------------------------
# Stage 5 lexical candidates
# --------------------------------------------------

lexical_map = {}


if LEXICAL_FILE.exists():

    with LEXICAL_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            key = pair_key(
                row["sample_id_1"],
                row["sample_id_2"],
            )

            lexical_map[
                key
            ] = row


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

print("=" * 78)
print("STAGE 6B - SEMANTIC CANDIDATES & CLUSTERS")
print("=" * 78)

print(
    f"Toplam kayıt             : "
    f"{len(records)}"
)

print(
    f"Semantic duplicate P99.5 : "
    f"{DUPLICATE_THRESHOLD:.6f}"
)

print(
    f"High priority P99.9      : "
    f"{HIGH_PRIORITY_THRESHOLD:.6f}"
)

print(
    f"Lexical candidate sayısı : "
    f"{len(lexical_map)}"
)


print("\nEmbedding modeli yükleniyor...")


model = SentenceTransformer(
    MODEL_NAME
)


texts = [
    "query: "
    + record["complaint_text"]
    for record in records
]


print("Embedding hesaplanıyor...")


embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype(
    np.float32
)


np.save(
    EMBEDDING_FILE,
    embeddings,
)


similarity = (
    embeddings
    @ embeddings.T
).astype(
    np.float32
)


# --------------------------------------------------
# Pair scanning
# --------------------------------------------------

n = len(records)

duplicate_pairs = []

scenario_heap = []
category_heap = []

uf = UnionFind(
    n
)


for i in range(n):

    r1 = records[i]

    for j in range(
        i + 1,
        n,
    ):

        r2 = records[j]

        score = float(
            similarity[i, j]
        )

        same_category = (
            r1["category"]
            == r2["category"]
        )

        same_scenario = (
            r1["scenario"]
            == r2["scenario"]
        )

        key = pair_key(
            r1["sample_id"],
            r2["sample_id"],
        )

        lexical = (
            lexical_map.get(
                key
            )
        )


        # ------------------------------------------
        # Same category + same scenario
        # Duplicate candidate
        # ------------------------------------------

        if (
            same_category
            and same_scenario
        ):

            semantic_candidate = (
                score
                >= DUPLICATE_THRESHOLD
            )

            lexical_candidate = (
                lexical is not None
            )

            if not (
                semantic_candidate
                or lexical_candidate
            ):

                continue


            raw_exact = (
                r1["complaint_text"]
                == r2["complaint_text"]
            )

            normalized_exact = (
                r1["_normalized"]
                == r2["_normalized"]
            )


            if raw_exact:

                priority = (
                    "EXACT"
                )

            elif normalized_exact:

                priority = (
                    "NORMALIZED_EXACT"
                )

            elif (
                score
                >= HIGH_PRIORITY_THRESHOLD
            ):

                priority = (
                    "HIGH"
                )

            else:

                priority = (
                    "REVIEW"
                )


            if (
                semantic_candidate
                and lexical_candidate
            ):

                candidate_source = (
                    "SEMANTIC_AND_LEXICAL"
                )

            elif semantic_candidate:

                candidate_source = (
                    "SEMANTIC_ONLY"
                )

            else:

                candidate_source = (
                    "LEXICAL_ONLY"
                )


            row = {
                "_i":
                    i,

                "_j":
                    j,

                "sample_id_1":
                    r1["sample_id"],

                "sample_id_2":
                    r2["sample_id"],

                "category":
                    r1["category"],

                "scenario":
                    r1["scenario"],

                "generator_1":
                    r1["generator_id"],

                "generator_2":
                    r2["generator_id"],

                "batch_1":
                    r1["generation_batch"],

                "batch_2":
                    r2["generation_batch"],

                "difficulty_1":
                    r1["difficulty"],

                "difficulty_2":
                    r2["difficulty"],

                "style_1":
                    r1["language_style"],

                "style_2":
                    r2["language_style"],

                "has_typo_1":
                    r1["has_typo"],

                "has_typo_2":
                    r2["has_typo"],

                "semantic_similarity":
                    round(
                        score,
                        6,
                    ),

                "lexical_candidate":
                    lexical_candidate,

                "lexical_pair_type":
                    (
                        lexical[
                            "pair_type"
                        ]
                        if lexical
                        else ""
                    ),

                "char_similarity":
                    (
                        lexical[
                            "char_similarity"
                        ]
                        if lexical
                        else ""
                    ),

                "word_similarity":
                    (
                        lexical[
                            "word_similarity"
                        ]
                        if lexical
                        else ""
                    ),

                "candidate_source":
                    candidate_source,

                "priority":
                    priority,

                "complaint_text_1":
                    r1[
                        "complaint_text"
                    ],

                "complaint_text_2":
                    r2[
                        "complaint_text"
                    ],
            }


            duplicate_pairs.append(
                row
            )

            uf.union(
                i,
                j,
            )

            continue


        # ------------------------------------------
        # Same category, different scenario:
        # scenario-boundary audit
        # ------------------------------------------

        if same_category:

            push_top_k(
                scenario_heap,
                (
                    score,
                    i,
                    j,
                ),
                SCENARIO_BOUNDARY_TOP_K,
            )

            continue


        # ------------------------------------------
        # Different category:
        # label-boundary / hard-negative audit
        # ------------------------------------------

        push_top_k(
            category_heap,
            (
                score,
                i,
                j,
            ),
            CATEGORY_BOUNDARY_TOP_K,
        )


# --------------------------------------------------
# Duplicate candidate CSV
# --------------------------------------------------

duplicate_pairs.sort(
    key=lambda x: (
        x[
            "semantic_similarity"
        ],
        x[
            "lexical_candidate"
        ],
    ),
    reverse=True,
)


DUPLICATE_FIELDS = [
    "sample_id_1",
    "sample_id_2",
    "category",
    "scenario",
    "generator_1",
    "generator_2",
    "batch_1",
    "batch_2",
    "difficulty_1",
    "difficulty_2",
    "style_1",
    "style_2",
    "has_typo_1",
    "has_typo_2",
    "semantic_similarity",
    "lexical_candidate",
    "lexical_pair_type",
    "char_similarity",
    "word_similarity",
    "candidate_source",
    "priority",
    "complaint_text_1",
    "complaint_text_2",
]


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with DUPLICATE_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=DUPLICATE_FIELDS,
    )

    writer.writeheader()

    for row in duplicate_pairs:

        writer.writerow({
            field:
                row[field]
            for field
            in DUPLICATE_FIELDS
        })


# --------------------------------------------------
# Semantic duplicate clusters
# --------------------------------------------------

cluster_map = defaultdict(
    list
)


for i in range(n):

    root = uf.find(
        i
    )

    cluster_map[
        root
    ].append(
        i
    )


clusters = [
    indexes
    for indexes
    in cluster_map.values()
    if len(indexes) > 1
]


clusters.sort(
    key=len,
    reverse=True,
)


record_to_cluster = {}


for cluster_id, indexes in enumerate(
    clusters,
    start=1,
):

    for index in indexes:

        record_to_cluster[
            index
        ] = cluster_id


cluster_edges = defaultdict(
    list
)


for pair in duplicate_pairs:

    i = pair["_i"]
    j = pair["_j"]

    cluster_id = (
        record_to_cluster[
            i
        ]
    )

    cluster_edges[
        cluster_id
    ].append({
        "sample_id_1":
            pair[
                "sample_id_1"
            ],

        "sample_id_2":
            pair[
                "sample_id_2"
            ],

        "semantic_similarity":
            pair[
                "semantic_similarity"
            ],

        "candidate_source":
            pair[
                "candidate_source"
            ],

        "priority":
            pair[
                "priority"
            ],
    })


with CLUSTER_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for cluster_id, indexes in enumerate(
        clusters,
        start=1,
    ):

        members = []

        for index in indexes:

            r = records[index]

            members.append({
                "sample_id":
                    r["sample_id"],

                "category":
                    r["category"],

                "scenario":
                    r["scenario"],

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

                "complaint_text":
                    r[
                        "complaint_text"
                    ],
            })


        edges = (
            cluster_edges[
                cluster_id
            ]
        )


        similarities = [
            edge[
                "semantic_similarity"
            ]
            for edge in edges
        ]


        item = {
            "cluster_id":
                cluster_id,

            "member_count":
                len(members),

            "category":
                members[0][
                    "category"
                ],

            "scenario":
                members[0][
                    "scenario"
                ],

            "candidate_edge_count":
                len(edges),

            "max_candidate_similarity":
                max(
                    similarities
                ),

            "min_candidate_similarity":
                min(
                    similarities
                ),

            "members":
                members,

            "candidate_edges":
                edges,
        }


        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


# --------------------------------------------------
# Boundary rows
# --------------------------------------------------

def build_boundary_rows(
    heap,
    relation,
):

    ranked = sorted(
        heap,
        key=lambda x: x[0],
        reverse=True,
    )

    rows = []

    for rank, (
        score,
        i,
        j,
    ) in enumerate(
        ranked,
        start=1,
    ):

        r1 = records[i]
        r2 = records[j]

        lexical = lexical_map.get(
            pair_key(
                r1["sample_id"],
                r2["sample_id"],
            )
        )

        rows.append({
            "rank":
                rank,

            "relation":
                relation,

            "semantic_similarity":
                round(
                    float(score),
                    6,
                ),

            "sample_id_1":
                r1["sample_id"],

            "sample_id_2":
                r2["sample_id"],

            "category_1":
                r1["category"],

            "category_2":
                r2["category"],

            "scenario_1":
                r1["scenario"],

            "scenario_2":
                r2["scenario"],

            "generator_1":
                r1["generator_id"],

            "generator_2":
                r2["generator_id"],

            "batch_1":
                r1[
                    "generation_batch"
                ],

            "batch_2":
                r2[
                    "generation_batch"
                ],

            "lexical_candidate":
                (
                    lexical is not None
                ),

            "complaint_text_1":
                r1[
                    "complaint_text"
                ],

            "complaint_text_2":
                r2[
                    "complaint_text"
                ],
        })

    return rows


scenario_rows = build_boundary_rows(
    scenario_heap,
    "SAME_CATEGORY_DIFF_SCENARIO",
)

category_rows = build_boundary_rows(
    category_heap,
    "DIFFERENT_CATEGORY",
)


BOUNDARY_FIELDS = [
    "rank",
    "relation",
    "semantic_similarity",
    "sample_id_1",
    "sample_id_2",
    "category_1",
    "category_2",
    "scenario_1",
    "scenario_2",
    "generator_1",
    "generator_2",
    "batch_1",
    "batch_2",
    "lexical_candidate",
    "complaint_text_1",
    "complaint_text_2",
]


for path, rows in [
    (
        SCENARIO_BOUNDARY_FILE,
        scenario_rows,
    ),
    (
        CATEGORY_BOUNDARY_FILE,
        category_rows,
    ),
]:

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=BOUNDARY_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

lexical_overlap_count = sum(
    1
    for pair in duplicate_pairs
    if pair[
        "lexical_candidate"
    ]
)


semantic_only_count = sum(
    1
    for pair in duplicate_pairs
    if pair[
        "candidate_source"
    ]
    == "SEMANTIC_ONLY"
)


high_priority_count = sum(
    1
    for pair in duplicate_pairs
    if pair[
        "priority"
    ]
    in {
        "EXACT",
        "NORMALIZED_EXACT",
        "HIGH",
    }
)


records_in_clusters = sum(
    len(cluster)
    for cluster in clusters
)


summary = {
    "input_file":
        str(INPUT_FILE),

    "input_sha256":
        sha256_file(
            INPUT_FILE
        ),

    "model":
        MODEL_NAME,

    "record_count":
        n,

    "embedding_dimension":
        int(
            embeddings.shape[1]
        ),

    "duplicate_candidate_policy": {
        "empirical_threshold_source":
            (
                "Stage 6A "
                "SAME_CATEGORY_SAME_SCENARIO "
                "distribution"
            ),

        "candidate_threshold_p99_5":
            DUPLICATE_THRESHOLD,

        "high_priority_threshold_p99_9":
            HIGH_PRIORITY_THRESHOLD,

        "lexical_candidates_always_included":
            True,

        "automatic_deletion":
            False,
    },

    "duplicate_candidate_pair_count":
        len(
            duplicate_pairs
        ),

    "high_priority_pair_count":
        high_priority_count,

    "lexical_overlap_pair_count":
        lexical_overlap_count,

    "semantic_only_pair_count":
        semantic_only_count,

    "semantic_cluster_count":
        len(
            clusters
        ),

    "records_in_semantic_clusters":
        records_in_clusters,

    "largest_cluster_size":
        max(
            (
                len(cluster)
                for cluster in clusters
            ),
            default=0,
        ),

    "scenario_boundary_top_k":
        SCENARIO_BOUNDARY_TOP_K,

    "cross_category_boundary_top_k":
        CATEGORY_BOUNDARY_TOP_K,

    "policy":
        (
            "High semantic similarity is not treated "
            "as automatic evidence of duplication. "
            "Same-category/same-scenario pairs form "
            "duplicate-review candidates. "
            "Cross-scenario and cross-category pairs "
            "are retained as boundary-review candidates "
            "because semantic overlap may represent "
            "valuable hard negatives."
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

print("\n" + "=" * 78)
print("STAGE 6B SONUÇ")
print("=" * 78)

print(
    f"Duplicate candidate pair  : "
    f"{len(duplicate_pairs)}"
)

print(
    f"High priority pair        : "
    f"{high_priority_count}"
)

print(
    f"Lexical + semantic overlap: "
    f"{lexical_overlap_count}"
)

print(
    f"Semantic-only pair        : "
    f"{semantic_only_count}"
)

print(
    f"Semantic cluster          : "
    f"{len(clusters)}"
)

print(
    f"Cluster içindeki kayıt    : "
    f"{records_in_clusters}"
)

print(
    f"En büyük cluster          : "
    f"{summary['largest_cluster_size']}"
)

print(
    f"Scenario boundary queue   : "
    f"{len(scenario_rows)}"
)

print(
    f"Cross-category queue      : "
    f"{len(category_rows)}"
)


print("\nEn yüksek 15 duplicate aday:")

for pair in duplicate_pairs[:15]:

    print(
        f"  "
        f"{pair['semantic_similarity']:.6f}  "
        f"{pair['sample_id_1']}  <->  "
        f"{pair['sample_id_2']}  "
        f"[{pair['candidate_source']} / "
        f"{pair['priority']}]"
    )


print("\nDosyalar:")

print(
    f"  Duplicate : {DUPLICATE_FILE}"
)

print(
    f"  Clusters  : {CLUSTER_FILE}"
)

print(
    f"  Scenario  : {SCENARIO_BOUNDARY_FILE}"
)

print(
    f"  Category  : {CATEGORY_BOUNDARY_FILE}"
)

print(
    f"  Summary   : {SUMMARY_FILE}"
)

print(
    f"  Embedding : {EMBEDDING_FILE}"
)