import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

EXACT_FILE = REPORT_DIR / (
    "05_exact_duplicate_groups.jsonl"
)

PAIR_FILE = REPORT_DIR / (
    "05_lexical_candidate_pairs.csv"
)

CLUSTER_FILE = REPORT_DIR / (
    "05_lexical_clusters.jsonl"
)

SUMMARY_FILE = REPORT_DIR / (
    "05_lexical_summary.json"
)


# --------------------------------------------------
# Candidate thresholds
#
# Bunlar otomatik reject kriteri DEĞİLDİR.
# Yalnızca review adayı üretir.
# --------------------------------------------------

THRESHOLDS = {
    "same_category_same_scenario": {
        "char": 0.74,
        "word": 0.68,
    },

    "same_category_diff_scenario": {
        "char": 0.82,
        "word": 0.76,
    },

    "different_category": {
        "char": 0.88,
        "word": 0.84,
    },

    # Cluster'a alınabilecek daha güçlü
    # same-category/same-scenario lexical çiftler
    "strong_same_scenario": {
        "char": 0.84,
        "word": 0.80,
    },
}


# --------------------------------------------------
# Normalization
# --------------------------------------------------

def normalize_text(text):
    """
    Conservative lexical normalization.

    - Unicode NFKC
    - Turkish-aware I/İ normalization
    - case normalization
    - apostrophe normalization
    - punctuation removal
    - whitespace collapse

    Yazım hatalarını veya kelimeleri semantik olarak
    değiştirmiyoruz.
    """

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

    text = text.replace(
        "_",
        " ",
    )

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

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

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
                root_b,
                root_a,
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

        obj = json.loads(
            line
        )

        obj["_normalized_text"] = (
            normalize_text(
                obj["complaint_text"]
            )
        )

        records.append(
            obj
        )


if len(records) != 1750:

    raise ValueError(
        f"Beklenen 1750 kayıt, "
        f"bulunan {len(records)}."
    )


n = len(records)

print("=" * 78)
print("STAGE 5 - EXACT + LEXICAL DUPLICATE QA")
print("=" * 78)

print(
    f"Toplam kayıt : {n}"
)

print(
    f"Toplam çift  : "
    f"{n * (n - 1) // 2}"
)


# --------------------------------------------------
# Exact duplicate grupları
# --------------------------------------------------

raw_groups = defaultdict(
    list
)

normalized_groups = defaultdict(
    list
)


for index, record in enumerate(
    records
):

    raw_groups[
        record["complaint_text"]
    ].append(index)

    normalized_groups[
        record["_normalized_text"]
    ].append(index)


raw_duplicate_groups = [
    indexes
    for indexes in raw_groups.values()
    if len(indexes) > 1
]


normalized_duplicate_groups = [
    indexes
    for indexes in normalized_groups.values()
    if len(indexes) > 1
]


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with EXACT_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    group_number = 0

    for duplicate_type, groups in [
        (
            "RAW_EXACT",
            raw_duplicate_groups,
        ),
        (
            "NORMALIZED_EXACT",
            normalized_duplicate_groups,
        ),
    ]:

        for indexes in groups:

            group_number += 1

            members = []

            for i in indexes:

                r = records[i]

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
                        r["generation_batch"],

                    "complaint_text":
                        r["complaint_text"],
                })

            item = {
                "group_id":
                    group_number,

                "duplicate_type":
                    duplicate_type,

                "member_count":
                    len(members),

                "members":
                    members,
            }

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

normalized_texts = [
    r["_normalized_text"]
    for r in records
]


print("\nCharacter TF-IDF hazırlanıyor...")


char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    lowercase=False,
    sublinear_tf=True,
    min_df=1,
    norm="l2",
)


char_matrix = (
    char_vectorizer.fit_transform(
        normalized_texts
    )
)


print(
    f"  feature sayısı: "
    f"{char_matrix.shape[1]}"
)


print("\nWord TF-IDF hazırlanıyor...")


word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    lowercase=False,
    sublinear_tf=True,
    min_df=1,
    norm="l2",
    token_pattern=r"(?u)\b\w+\b",
)


word_matrix = (
    word_vectorizer.fit_transform(
        normalized_texts
    )
)


print(
    f"  feature sayısı: "
    f"{word_matrix.shape[1]}"
)


# --------------------------------------------------
# Cosine matrices
# --------------------------------------------------

print("\nCosine similarity hesaplanıyor...")


char_similarity = cosine_similarity(
    char_matrix
).astype(
    np.float32
)


word_similarity = cosine_similarity(
    word_matrix
).astype(
    np.float32
)


# --------------------------------------------------
# Pair candidate üretimi
# --------------------------------------------------

candidate_pairs = []

pair_type_counts = Counter()
relation_counts = Counter()

union_find = UnionFind(
    n
)


for i in range(n):

    r1 = records[i]

    for j in range(
        i + 1,
        n,
    ):

        r2 = records[j]

        char_score = float(
            char_similarity[i, j]
        )

        word_score = float(
            word_similarity[i, j]
        )

        raw_exact = (
            r1["complaint_text"]
            == r2["complaint_text"]
        )

        normalized_exact = (
            r1["_normalized_text"]
            == r2["_normalized_text"]
        )


        same_category = (
            r1["category"]
            == r2["category"]
        )

        same_scenario = (
            r1["scenario"]
            == r2["scenario"]
        )


        # ------------------------------------------
        # Pair relation
        # ------------------------------------------

        if (
            same_category
            and same_scenario
        ):

            relation = (
                "SAME_CATEGORY_SAME_SCENARIO"
            )

            threshold = THRESHOLDS[
                "same_category_same_scenario"
            ]

        elif same_category:

            relation = (
                "SAME_CATEGORY_DIFF_SCENARIO"
            )

            threshold = THRESHOLDS[
                "same_category_diff_scenario"
            ]

        else:

            relation = (
                "DIFFERENT_CATEGORY"
            )

            threshold = THRESHOLDS[
                "different_category"
            ]


        # ------------------------------------------
        # Candidate?
        # ------------------------------------------

        is_candidate = (
            raw_exact
            or normalized_exact
            or char_score
            >= threshold["char"]
            or word_score
            >= threshold["word"]
        )


        if not is_candidate:
            continue


        # ------------------------------------------
        # Pair type
        # ------------------------------------------

        if raw_exact:

            pair_type = (
                "RAW_EXACT"
            )

        elif normalized_exact:

            pair_type = (
                "NORMALIZED_EXACT"
            )

        elif not same_category:

            pair_type = (
                "CROSS_CATEGORY_HIGH_SIMILARITY"
            )

        elif not same_scenario:

            pair_type = (
                "CROSS_SCENARIO_HIGH_SIMILARITY"
            )

        else:

            strong_threshold = (
                THRESHOLDS[
                    "strong_same_scenario"
                ]
            )

            if (
                char_score
                >= strong_threshold["char"]
                or word_score
                >= strong_threshold["word"]
            ):

                pair_type = (
                    "STRONG_LEXICAL_NEAR_DUPLICATE"
                )

            else:

                pair_type = (
                    "LEXICAL_REVIEW_CANDIDATE"
                )


        max_score = max(
            char_score,
            word_score,
        )


        candidate = {
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
                r1["generation_batch"],

            "batch_2":
                r2["generation_batch"],

            "same_generator":
                (
                    r1["generator_id"]
                    == r2["generator_id"]
                ),

            "same_batch":
                (
                    r1["generation_batch"]
                    == r2["generation_batch"]
                ),

            "relation":
                relation,

            "pair_type":
                pair_type,

            "char_similarity":
                round(
                    char_score,
                    6,
                ),

            "word_similarity":
                round(
                    word_score,
                    6,
                ),

            "max_similarity":
                round(
                    max_score,
                    6,
                ),

            "complaint_text_1":
                r1["complaint_text"],

            "complaint_text_2":
                r2["complaint_text"],
        }


        candidate_pairs.append(
            candidate
        )

        pair_type_counts[
            pair_type
        ] += 1

        relation_counts[
            relation
        ] += 1


        # ------------------------------------------
        # Conservative lexical clusters
        #
        # Yalnızca aynı category + scenario içindeki
        # güçlü çiftleri cluster ediyoruz.
        # ------------------------------------------

        if (
            same_category
            and same_scenario
            and pair_type
            in {
                "RAW_EXACT",
                "NORMALIZED_EXACT",
                "STRONG_LEXICAL_NEAR_DUPLICATE",
            }
        ):

            union_find.union(
                i,
                j,
            )


# --------------------------------------------------
# Candidate CSV
# --------------------------------------------------

candidate_pairs.sort(
    key=lambda x: (
        x["max_similarity"],
        x["char_similarity"],
        x["word_similarity"],
    ),
    reverse=True,
)


pair_fields = [
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
    "same_generator",
    "same_batch",
    "relation",
    "pair_type",
    "char_similarity",
    "word_similarity",
    "max_similarity",
    "complaint_text_1",
    "complaint_text_2",
]


with PAIR_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=pair_fields,
    )

    writer.writeheader()

    writer.writerows(
        candidate_pairs
    )


# --------------------------------------------------
# Cluster üretimi
# --------------------------------------------------

cluster_members = defaultdict(
    list
)


for i in range(n):

    root = union_find.find(
        i
    )

    cluster_members[
        root
    ].append(i)


clusters = [
    indexes
    for indexes
    in cluster_members.values()
    if len(indexes) > 1
]


clusters.sort(
    key=len,
    reverse=True,
)


with CLUSTER_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for cluster_id, indexes in enumerate(
        clusters,
        start=1,
    ):

        members = []

        for i in indexes:

            r = records[i]

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
                    r["generation_batch"],

                "complaint_text":
                    r["complaint_text"],
            })


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

            "members":
                members,
        }


        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

cross_category_pairs = sum(
    1
    for p in candidate_pairs
    if p["relation"]
    == "DIFFERENT_CATEGORY"
)


cross_scenario_pairs = sum(
    1
    for p in candidate_pairs
    if p["relation"]
    == "SAME_CATEGORY_DIFF_SCENARIO"
)


strong_pairs = sum(
    1
    for p in candidate_pairs
    if p["pair_type"]
    == "STRONG_LEXICAL_NEAR_DUPLICATE"
)


cluster_member_count = sum(
    len(cluster)
    for cluster in clusters
)


summary = {
    "total_records":
        n,

    "total_possible_pairs":
        n * (n - 1) // 2,

    "raw_exact_group_count":
        len(
            raw_duplicate_groups
        ),

    "normalized_exact_group_count":
        len(
            normalized_duplicate_groups
        ),

    "lexical_candidate_pair_count":
        len(
            candidate_pairs
        ),

    "strong_same_scenario_pair_count":
        strong_pairs,

    "cross_scenario_candidate_count":
        cross_scenario_pairs,

    "cross_category_candidate_count":
        cross_category_pairs,

    "pair_type_counts":
        dict(
            pair_type_counts
        ),

    "relation_counts":
        dict(
            relation_counts
        ),

    "cluster_count":
        len(clusters),

    "records_in_clusters":
        cluster_member_count,

    "largest_cluster_size":
        (
            max(
                (
                    len(c)
                    for c in clusters
                ),
                default=0,
            )
        ),

    "thresholds":
        THRESHOLDS,

    "policy": (
        "Lexical similarity thresholds are candidate-generation "
        "heuristics only. No record is automatically removed. "
        "Final duplicate decisions require semantic review and "
        "will later be supplemented by embedding-based semantic "
        "similarity analysis."
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
print("LEXICAL QA SONUCU")
print("=" * 78)

print(
    f"Raw exact group             : "
    f"{len(raw_duplicate_groups)}"
)

print(
    f"Normalized exact group      : "
    f"{len(normalized_duplicate_groups)}"
)

print(
    f"Lexical candidate pair      : "
    f"{len(candidate_pairs)}"
)

print(
    f"Strong same-scenario pair   : "
    f"{strong_pairs}"
)

print(
    f"Cross-scenario candidate    : "
    f"{cross_scenario_pairs}"
)

print(
    f"Cross-category candidate    : "
    f"{cross_category_pairs}"
)

print(
    f"Lexical cluster             : "
    f"{len(clusters)}"
)

print(
    f"Cluster içindeki kayıt      : "
    f"{cluster_member_count}"
)

print(
    f"En büyük cluster            : "
    f"{summary['largest_cluster_size']}"
)


print("\nPair türleri:")

for key, value in sorted(
    pair_type_counts.items()
):

    print(
        f"  {key:<35}: {value}"
    )


print("\nEn yüksek 10 lexical aday:")

for pair in candidate_pairs[:10]:

    print(
        f"  "
        f"{pair['max_similarity']:.4f}  "
        f"{pair['sample_id_1']}  <->  "
        f"{pair['sample_id_2']}  "
        f"[{pair['pair_type']}]"
    )


print("\nDosyalar:")

print(
    f"  Exact    : {EXACT_FILE}"
)

print(
    f"  Pairs    : {PAIR_FILE}"
)

print(
    f"  Clusters : {CLUSTER_FILE}"
)

print(
    f"  Summary  : {SUMMARY_FILE}"
)