import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

SUMMARY_FILE = REPORT_DIR / (
    "06a_semantic_similarity_summary.json"
)

TOP_PAIRS_FILE = REPORT_DIR / (
    "06a_semantic_top_pairs.csv"
)


MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K_PER_RELATION = 200


def percentile(values, p):

    if len(values) == 0:
        return None

    return float(
        np.percentile(
            values,
            p,
        )
    )


def stats(values):

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    return {
        "count": int(len(values)),
        "mean": round(float(values.mean()), 6),
        "median": round(percentile(values, 50), 6),
        "p90": round(percentile(values, 90), 6),
        "p95": round(percentile(values, 95), 6),
        "p99": round(percentile(values, 99), 6),
        "p99_5": round(percentile(values, 99.5), 6),
        "p99_9": round(percentile(values, 99.9), 6),
        "max": round(float(values.max()), 6),
    }


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

        records.append(
            json.loads(line)
        )


if len(records) != 1750:

    raise ValueError(
        f"Beklenen 1750 kayıt, "
        f"bulunan {len(records)}"
    )


print("=" * 78)
print("STAGE 6A - SEMANTIC SIMILARITY PROFILE")
print("=" * 78)

print(
    f"Toplam kayıt : {len(records)}"
)

print(
    f"Model        : {MODEL_NAME}"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

print("\nEmbedding modeli yükleniyor...")

model = SentenceTransformer(
    MODEL_NAME
)


# E5 ailesinde retrieval/semantic embedding kullanımında
# prefix kullanmak önerilir.
texts = [
    "query: " + r["complaint_text"]
    for r in records
]


print("Embedding hesaplanıyor...")


embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)


print(
    f"Embedding shape: {embeddings.shape}"
)


# Normalize edildiği için dot product = cosine similarity.
similarity = (
    embeddings @ embeddings.T
).astype(
    np.float32
)


# --------------------------------------------------
# Relation bazında bütün çiftleri topla
# --------------------------------------------------

relation_scores = {
    "SAME_CATEGORY_SAME_SCENARIO": [],
    "SAME_CATEGORY_DIFF_SCENARIO": [],
    "DIFFERENT_CATEGORY": [],
}


top_pairs = {
    "SAME_CATEGORY_SAME_SCENARIO": [],
    "SAME_CATEGORY_DIFF_SCENARIO": [],
    "DIFFERENT_CATEGORY": [],
}


n = len(records)


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


        if (
            r1["category"]
            == r2["category"]
            and r1["scenario"]
            == r2["scenario"]
        ):

            relation = (
                "SAME_CATEGORY_SAME_SCENARIO"
            )

        elif (
            r1["category"]
            == r2["category"]
        ):

            relation = (
                "SAME_CATEGORY_DIFF_SCENARIO"
            )

        else:

            relation = (
                "DIFFERENT_CATEGORY"
            )


        relation_scores[
            relation
        ].append(
            score
        )


        top_pairs[
            relation
        ].append({
            "score":
                score,

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

            "complaint_text_1":
                r1["complaint_text"],

            "complaint_text_2":
                r2["complaint_text"],
        })


# --------------------------------------------------
# Sadece en yüksek N çifti sakla
# --------------------------------------------------

selected_pairs = []


for relation in top_pairs:

    pairs = sorted(
        top_pairs[relation],
        key=lambda x: x["score"],
        reverse=True,
    )[:TOP_K_PER_RELATION]


    for rank, pair in enumerate(
        pairs,
        start=1,
    ):

        selected_pairs.append({
            "relation":
                relation,

            "rank":
                rank,

            "semantic_similarity":
                round(
                    pair["score"],
                    6,
                ),

            **{
                k: v
                for k, v
                in pair.items()
                if k != "score"
            },
        })


# --------------------------------------------------
# CSV
# --------------------------------------------------

fields = [
    "relation",
    "rank",
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
    "complaint_text_1",
    "complaint_text_2",
]


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with TOP_PAIRS_FILE.open(
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
        selected_pairs
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

relation_summary = {
    relation:
        stats(scores)
    for relation, scores
    in relation_scores.items()
}


summary = {
    "model":
        MODEL_NAME,

    "record_count":
        n,

    "embedding_dimension":
        int(
            embeddings.shape[1]
        ),

    "total_possible_pairs":
        n * (n - 1) // 2,

    "relation_statistics":
        relation_summary,

    "top_k_per_relation":
        TOP_K_PER_RELATION,

    "policy": (
        "No fixed semantic duplicate threshold is "
        "applied at this stage. Similarity distributions "
        "are measured first. Candidate thresholds and "
        "clusters will be determined only after inspecting "
        "the empirical score distribution."
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
print("SEMANTIC SIMILARITY DAĞILIMI")
print("=" * 78)


for relation, result in (
    relation_summary.items()
):

    print(
        f"\n{relation}"
    )

    for key, value in (
        result.items()
    ):

        print(
            f"  {key:<8}: {value}"
        )


print("\nHer relation için en yüksek 10 çift:")


for relation in top_pairs:

    print(
        f"\n{relation}"
    )

    pairs = sorted(
        top_pairs[relation],
        key=lambda x: x["score"],
        reverse=True,
    )[:10]

    for p in pairs:

        print(
            f"  {p['score']:.6f}  "
            f"{p['sample_id_1']}  <->  "
            f"{p['sample_id_2']}"
        )


print("\nDosyalar:")

print(
    f"  Summary   : {SUMMARY_FILE}"
)

print(
    f"  Top pairs : {TOP_PAIRS_FILE}"
)