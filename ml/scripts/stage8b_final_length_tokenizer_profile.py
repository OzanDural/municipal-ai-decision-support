import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


FINAL_FILE = Path(
    "data/processed/master_final.jsonl"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

SUMMARY_FILE = REPORT_DIR / (
    "08b_final_length_tokenizer_summary.json"
)

PROFILE_CSV = REPORT_DIR / (
    "08b_final_length_tokenizer_profile.csv"
)

LONGEST_CSV = REPORT_DIR / (
    "08b_longest_records.csv"
)


EXPECTED_FINAL_SHA256 = (
    "c6d8a06e0fb757c3b1db11a87959f7086fd4b22da5c97268522c5fd4d77ed1c7"
)

TOKENIZER_NAME = (
    "dbmdz/bert-base-turkish-cased"
)

CANDIDATE_MAX_LENGTHS = [
    64,
    96,
    128,
    160,
    192,
    256,
]


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


def percentile_dict(values):

    arr = np.asarray(
        values,
        dtype=float,
    )

    return {
        "min":
            float(np.min(arr)),

        "mean":
            float(np.mean(arr)),

        "median":
            float(np.median(arr)),

        "p75":
            float(
                np.percentile(
                    arr,
                    75,
                )
            ),

        "p90":
            float(
                np.percentile(
                    arr,
                    90,
                )
            ),

        "p95":
            float(
                np.percentile(
                    arr,
                    95,
                )
            ),

        "p99":
            float(
                np.percentile(
                    arr,
                    99,
                )
            ),

        "max":
            float(np.max(arr)),
    }


def sentence_count(text):

    # YaklaÅŸÄ±k cÃ¼mle sayÄ±sÄ±.
    # Dataset QA amaÃ§lÄ±dÄ±r; linguistic
    # sentence segmentation deÄŸildir.
    parts = re.split(
        r"[.!?]+",
        text.strip(),
    )

    parts = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    return max(
        1,
        len(parts),
    )


def source_family(record):

    batch = record[
        "generation_batch"
    ]

    if batch.startswith(
        "long_"
    ):
        return "LONG"

    return "ORIGINAL"


# ==================================================
# Verify frozen final dataset
# ==================================================

actual_hash = sha256_file(
    FINAL_FILE
)


if actual_hash != EXPECTED_FINAL_SHA256:

    raise ValueError(
        "master_final.jsonl hash deÄŸiÅŸmiÅŸ!\n"
        f"Beklenen : {EXPECTED_FINAL_SHA256}\n"
        f"Bulunan  : {actual_hash}\n"
        "Profil oluÅŸturulmadÄ±."
    )


# ==================================================
# Load final records
# ==================================================

records = []

with FINAL_FILE.open(
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

        records.append(
            record
        )


if len(records) != 1667:

    raise ValueError(
        f"Beklenen final kayÄ±t 1667, "
        f"bulunan {len(records)}."
    )


if len({
    r["sample_id"]
    for r in records
}) != 1667:

    raise ValueError(
        "Final dataset iÃ§inde duplicate "
        "sample_id var."
    )


qa_status_counts = Counter(
    r["qa_status"]
    for r in records
)


if qa_status_counts != {
    "ACCEPTED": 1667
}:

    raise ValueError(
        "Final dataset qa_status hatalÄ±: "
        f"{dict(qa_status_counts)}"
    )


# ==================================================
# Load tokenizer
# ==================================================

print(
    f"Tokenizer yÃ¼kleniyor: "
    f"{TOKENIZER_NAME}"
)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    use_fast=True,
)


# ==================================================
# Record-level profile
# ==================================================

profile_rows = []


for index, record in enumerate(
    records,
    start=1,
):

    text = record[
        "complaint_text"
    ]


    word_count = len(
        text.split()
    )


    char_count = len(
        text
    )


    sent_count = sentence_count(
        text
    )


    # add_special_tokens=True:
    # gerÃ§ek BERT inputuna [CLS] + [SEP]
    # dahil edilmiÅŸ token uzunluÄŸu.
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False,
    )


    token_count = len(
        token_ids
    )


    profile_rows.append({
        "sample_id":
            record["sample_id"],

        "category":
            record["category"],

        "scenario":
            record["scenario"],

        "difficulty":
            record["difficulty"],

        "language_style":
            record["language_style"],

        "generator_id":
            record["generator_id"],

        "generation_batch":
            record["generation_batch"],

        "source_family":
            source_family(
                record
            ),

        "word_count":
            word_count,

        "char_count":
            char_count,

        "sentence_count":
            sent_count,

        "berturk_token_count":
            token_count,

        "complaint_text":
            text,
    })


# ==================================================
# Global descriptive statistics
# ==================================================

word_counts = [
    row["word_count"]
    for row in profile_rows
]

char_counts = [
    row["char_count"]
    for row in profile_rows
]

sentence_counts = [
    row["sentence_count"]
    for row in profile_rows
]

token_counts = [
    row["berturk_token_count"]
    for row in profile_rows
]


global_stats = {
    "word_count":
        percentile_dict(
            word_counts
        ),

    "char_count":
        percentile_dict(
            char_counts
        ),

    "sentence_count":
        percentile_dict(
            sentence_counts
        ),

    "berturk_token_count":
        percentile_dict(
            token_counts
        ),
}


# ==================================================
# Original vs long
# ==================================================

family_stats = {}


for family in [
    "ORIGINAL",
    "LONG",
]:

    subset = [
        row
        for row in profile_rows
        if row[
            "source_family"
        ] == family
    ]


    family_stats[
        family
    ] = {
        "count":
            len(subset),

        "word_count":
            percentile_dict([
                row["word_count"]
                for row in subset
            ]),

        "berturk_token_count":
            percentile_dict([
                row[
                    "berturk_token_count"
                ]
                for row in subset
            ]),
    }


# ==================================================
# Category tokenizer profile
# ==================================================

category_stats = {}


categories = sorted({
    row["category"]
    for row in profile_rows
})


for category in categories:

    subset = [
        row
        for row in profile_rows
        if row["category"]
        == category
    ]


    category_stats[
        category
    ] = {
        "count":
            len(subset),

        "word_count":
            percentile_dict([
                row["word_count"]
                for row in subset
            ]),

        "berturk_token_count":
            percentile_dict([
                row[
                    "berturk_token_count"
                ]
                for row in subset
            ]),
    }


# ==================================================
# Candidate max_length truncation profile
# ==================================================

truncation_profile = {}


for max_length in (
    CANDIDATE_MAX_LENGTHS
):

    over = [
        row
        for row in profile_rows
        if row[
            "berturk_token_count"
        ] > max_length
    ]


    equal_or_under = (
        len(profile_rows)
        - len(over)
    )


    truncation_profile[
        str(max_length)
    ] = {
        "records_over_limit":
            len(over),

        "records_not_truncated":
            equal_or_under,

        "truncation_rate":
            (
                len(over)
                / len(profile_rows)
            ),
    }


# ==================================================
# Token length bands
# ==================================================

def token_band(n):

    if n <= 32:
        return "001_032"

    if n <= 64:
        return "033_064"

    if n <= 96:
        return "065_096"

    if n <= 128:
        return "097_128"

    if n <= 160:
        return "129_160"

    if n <= 192:
        return "161_192"

    if n <= 256:
        return "193_256"

    return "257_PLUS"


token_band_counts = Counter(
    token_band(
        row[
            "berturk_token_count"
        ]
    )
    for row in profile_rows
)


# ==================================================
# Word length bands
# ==================================================

def word_band(n):

    if n <= 24:
        return "UNDER_25"

    if n <= 50:
        return "25_50"

    if n <= 80:
        return "51_80"

    if n <= 120:
        return "81_120"

    return "OVER_120"


word_band_counts = Counter(
    word_band(
        row["word_count"]
    )
    for row in profile_rows
)


# ==================================================
# Token/word ratio
# ==================================================

token_word_ratios = [
    row["berturk_token_count"]
    / max(
        1,
        row["word_count"],
    )
    for row in profile_rows
]


token_word_ratio_stats = (
    percentile_dict(
        token_word_ratios
    )
)


# ==================================================
# Longest records
# ==================================================

longest_rows = sorted(
    profile_rows,
    key=lambda row:
        (
            row[
                "berturk_token_count"
            ],
            row[
                "word_count"
            ],
        ),
    reverse=True,
)[:50]


# ==================================================
# Write full profile CSV
# ==================================================

profile_fields = [
    "sample_id",
    "category",
    "scenario",
    "difficulty",
    "language_style",
    "generator_id",
    "generation_batch",
    "source_family",
    "word_count",
    "char_count",
    "sentence_count",
    "berturk_token_count",
    "complaint_text",
]


with PROFILE_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=profile_fields,
    )

    writer.writeheader()

    writer.writerows(
        profile_rows
    )


# ==================================================
# Write longest records
# ==================================================

with LONGEST_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=profile_fields,
    )

    writer.writeheader()

    writer.writerows(
        longest_rows
    )


# ==================================================
# Summary
# ==================================================

summary = {
    "status":
        "FINAL_LENGTH_TOKENIZER_PROFILE_COMPLETE",

    "dataset_record_count":
        len(records),

    "dataset_sha256":
        actual_hash,

    "tokenizer_name":
        TOKENIZER_NAME,

    "tokenizer_class":
        tokenizer.__class__.__name__,

    "special_tokens_included":
        True,

    "global_statistics":
        global_stats,

    "source_family_statistics":
        family_stats,

    "category_statistics":
        category_stats,

    "word_length_bands":
        dict(
            word_band_counts
        ),

    "berturk_token_length_bands":
        dict(
            token_band_counts
        ),

    "token_word_ratio":
        token_word_ratio_stats,

    "candidate_max_length_truncation":
        truncation_profile,

    "dataset_modified":
        False,

    "policy": (
        "Tokenizer profiling was performed on the "
        "frozen final accepted dataset before any "
        "train/validation/test split. Token counts "
        "include BERT special tokens. No max_length "
        "is selected automatically; truncation rates "
        "are reported for evidence-based selection."
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
# Terminal report
# ==================================================

print("=" * 78)
print("STAGE 8B - FINAL LENGTH + BERTURK TOKENIZER PROFILE")
print("=" * 78)

print(
    f"Final dataset        : "
    f"{len(records)}"
)

print(
    f"Dataset SHA256       : "
    f"{actual_hash}"
)

print(
    f"Tokenizer            : "
    f"{TOKENIZER_NAME}"
)


print("\nWord count:")

for key, value in (
    global_stats[
        "word_count"
    ].items()
):

    print(
        f"  {key:<8}: "
        f"{value:.2f}"
    )


print("\nBERTurk token count:")

for key, value in (
    global_stats[
        "berturk_token_count"
    ].items()
):

    print(
        f"  {key:<8}: "
        f"{value:.2f}"
    )


print("\nSource family:")

for family in [
    "ORIGINAL",
    "LONG",
]:

    stats = family_stats[
        family
    ]

    print(
        f"  {family:<10}: "
        f"n={stats['count']}, "
        f"token mean="
        f"{stats['berturk_token_count']['mean']:.2f}, "
        f"p95="
        f"{stats['berturk_token_count']['p95']:.2f}, "
        f"max="
        f"{stats['berturk_token_count']['max']:.0f}"
    )


print("\nCandidate max_length:")

for max_length in (
    CANDIDATE_MAX_LENGTHS
):

    result = (
        truncation_profile[
            str(max_length)
        ]
    )

    print(
        f"  {max_length:<4} -> "
        f"truncate "
        f"{result['records_over_limit']:>4} / "
        f"{len(records)} "
        f"("
        f"{result['truncation_rate'] * 100:.2f}%"
        f")"
    )


print("\nWord bands:")

for key in [
    "UNDER_25",
    "25_50",
    "51_80",
    "81_120",
    "OVER_120",
]:

    print(
        f"  {key:<12}: "
        f"{word_band_counts.get(key, 0)}"
    )


print("\nToken bands:")

for key in [
    "001_032",
    "033_064",
    "065_096",
    "097_128",
    "129_160",
    "161_192",
    "193_256",
    "257_PLUS",
]:

    print(
        f"  {key:<12}: "
        f"{token_band_counts.get(key, 0)}"
    )


print("\nOutputs:")

print(
    f"  Profile : "
    f"{PROFILE_CSV}"
)

print(
    f"  Longest : "
    f"{LONGEST_CSV}"
)

print(
    f"  Summary : "
    f"{SUMMARY_FILE}"
)


print(
    "\nSTATUS: "
    "FINAL_LENGTH_TOKENIZER_PROFILE_COMPLETE"
)

