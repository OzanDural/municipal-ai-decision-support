import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

REPORT_DIR = Path(
    "ml/reports/final_test"
)

TFIDF_FILE = (
    REPORT_DIR
    / "tfidf_logreg"
    / "predictions.csv"
)

FROZEN_FILE = (
    REPORT_DIR
    / "frozen_berturk_linear_probe"
    / "predictions.csv"
)

FINETUNED_FILE = (
    REPORT_DIR
    / "full_finetuned_berturk"
    / "predictions.csv"
)


OUTPUT_DIR = (
    REPORT_DIR
    / "error_analysis"
)

FINETUNED_ERRORS_FILE = (
    OUTPUT_DIR
    / "13b_finetuned_errors_for_manual_review.csv"
)

MODEL_ERROR_SUMMARY_FILE = (
    OUTPUT_DIR
    / "13b_model_error_summary.csv"
)

CONFUSION_PAIRS_FILE = (
    OUTPUT_DIR
    / "13b_confusion_pairs.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "13b_error_analysis_summary.json"
)


# ============================================================
# VERIFY INPUTS
# ============================================================

required_files = [
    TFIDF_FILE,
    FROZEN_FILE,
    FINETUNED_FILE,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing Stage 13 output:\n{path}"
        )


# ============================================================
# LOAD
# ============================================================

tfidf = pd.read_csv(
    TFIDF_FILE
)

frozen = pd.read_csv(
    FROZEN_FILE
)

finetuned = pd.read_csv(
    FINETUNED_FILE
)


expected_count = 250


for name, df in [
    ("TFIDF", tfidf),
    ("FROZEN_BERTURK", frozen),
    ("FINETUNED_BERTURK", finetuned),
]:

    if len(df) != expected_count:

        raise RuntimeError(
            f"{name}: expected 250 rows, "
            f"found {len(df)}"
        )


# ============================================================
# NORMALIZE BOOLEAN COLUMN
# ============================================================

def normalize_correct_column(df):

    if df["correct"].dtype == object:

        df["correct"] = (
            df["correct"]
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False,
            })
        )

    if df["correct"].isna().any():

        raise RuntimeError(
            "correct column could not "
            "be parsed."
        )

    return df


tfidf = normalize_correct_column(
    tfidf
)

frozen = normalize_correct_column(
    frozen
)

finetuned = normalize_correct_column(
    finetuned
)


# ============================================================
# CHECK SAMPLE ALIGNMENT
# ============================================================

tfidf_ids = set(
    tfidf["sample_id"]
)

frozen_ids = set(
    frozen["sample_id"]
)

finetuned_ids = set(
    finetuned["sample_id"]
)


if not (
    tfidf_ids
    == frozen_ids
    == finetuned_ids
):

    raise RuntimeError(
        "Prediction files do not contain "
        "the same sample IDs."
    )


# ============================================================
# BUILD MODEL ERROR SUMMARY
# ============================================================

model_rows = []


for model_name, df in [
    (
        "TFIDF_LOGREG",
        tfidf,
    ),
    (
        "FROZEN_BERTURK_LINEAR_PROBE",
        frozen,
    ),
    (
        "FULL_FINETUNED_BERTURK",
        finetuned,
    ),
]:

    error_count = int(
        (~df["correct"]).sum()
    )

    model_rows.append({
        "model":
            model_name,

        "total":
            len(df),

        "correct":
            int(
                df["correct"].sum()
            ),

        "incorrect":
            error_count,

        "accuracy":
            float(
                df["correct"].mean()
            ),
    })


model_summary_df = pd.DataFrame(
    model_rows
)


# ============================================================
# FINETUNED ERRORS
# ============================================================

ft_errors = (
    finetuned[
        ~finetuned["correct"]
    ]
    .copy()
)


# ============================================================
# MERGE OTHER MODEL PREDICTIONS
# ============================================================

tfidf_compare = (
    tfidf[
        [
            "sample_id",
            "predicted_category",
            "correct",
            "top1_score",
            "top2_category",
            "top2_score",
            "score_margin",
        ]
    ]
    .rename(
        columns={
            "predicted_category":
                "tfidf_prediction",

            "correct":
                "tfidf_correct",

            "top1_score":
                "tfidf_top1_score",

            "top2_category":
                "tfidf_top2_category",

            "top2_score":
                "tfidf_top2_score",

            "score_margin":
                "tfidf_margin",
        }
    )
)


frozen_compare = (
    frozen[
        [
            "sample_id",
            "predicted_category",
            "correct",
            "top1_score",
            "top2_category",
            "top2_score",
            "score_margin",
        ]
    ]
    .rename(
        columns={
            "predicted_category":
                "frozen_prediction",

            "correct":
                "frozen_correct",

            "top1_score":
                "frozen_top1_score",

            "top2_category":
                "frozen_top2_category",

            "top2_score":
                "frozen_top2_score",

            "score_margin":
                "frozen_margin",
        }
    )
)


review_df = (
    ft_errors[
        [
            "sample_id",
            "complaint_text",
            "true_category",
            "predicted_category",
            "top1_score",
            "top2_category",
            "top2_score",
            "score_margin",
        ]
    ]
    .rename(
        columns={
            "predicted_category":
                "finetuned_prediction",

            "top1_score":
                "finetuned_top1_score",

            "top2_category":
                "finetuned_top2_category",

            "top2_score":
                "finetuned_top2_score",

            "score_margin":
                "finetuned_margin",
        }
    )
    .merge(
        tfidf_compare,
        on="sample_id",
        how="left",
    )
    .merge(
        frozen_compare,
        on="sample_id",
        how="left",
    )
)


# ============================================================
# ADD MANUAL REVIEW FIELDS
#
# Do NOT auto-invent an error cause.
# We will inspect these manually.
# ============================================================

review_df[
    "manual_error_type"
] = ""

review_df[
    "manual_notes"
] = ""

review_df[
    "label_suspected"
] = False


# ============================================================
# CONFUSION PAIRS FOR ALL MODELS
# ============================================================

confusion_rows = []


for model_name, df in [
    (
        "TFIDF_LOGREG",
        tfidf,
    ),
    (
        "FROZEN_BERTURK_LINEAR_PROBE",
        frozen,
    ),
    (
        "FULL_FINETUNED_BERTURK",
        finetuned,
    ),
]:

    errors = df[
        ~df["correct"]
    ]

    grouped = (
        errors
        .groupby(
            [
                "true_category",
                "predicted_category",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    for _, row in grouped.iterrows():

        confusion_rows.append({
            "model":
                model_name,

            "true_category":
                row[
                    "true_category"
                ],

            "predicted_category":
                row[
                    "predicted_category"
                ],

            "count":
                int(
                    row["count"]
                ),
        })


confusion_df = pd.DataFrame(
    confusion_rows
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


review_df.to_csv(
    FINETUNED_ERRORS_FILE,
    index=False,
)


model_summary_df.to_csv(
    MODEL_ERROR_SUMMARY_FILE,
    index=False,
)


confusion_df.to_csv(
    CONFUSION_PAIRS_FILE,
    index=False,
)


summary = {
    "status":
        "FINAL_TEST_ERROR_ANALYSIS_PREPARED",

    "test_size":
        250,

    "tfidf_errors":
        int(
            (~tfidf["correct"]).sum()
        ),

    "frozen_berturk_errors":
        int(
            (~frozen["correct"]).sum()
        ),

    "finetuned_berturk_errors":
        int(
            (~finetuned["correct"]).sum()
        ),

    "manual_review_required":
        len(review_df),

    "post_test_tuning_allowed":
        False,

    "note":
        (
            "This stage performs post-hoc "
            "error analysis only. No model, "
            "dataset, split, threshold or "
            "hyperparameter may be changed "
            "based on final-test errors."
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


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("=" * 80)
print(
    "STAGE 13B - FINAL TEST "
    "ERROR ANALYSIS"
)
print("=" * 80)

print()
print(
    "MODEL ERROR COUNTS"
)

for _, row in (
    model_summary_df.iterrows()
):

    print(
        f"{row['model']:<32} "
        f"errors="
        f"{int(row['incorrect']):>3} | "
        f"accuracy="
        f"{row['accuracy']:.6f}"
    )


print()
print(
    "FINE-TUNED BERTURK ERRORS"
)

print(
    f"Count: {len(review_df)}"
)


for index, row in (
    review_df.iterrows()
):

    print()
    print("-" * 80)

    print(
        f"Sample ID:"
        f" {row['sample_id']}"
    )

    print(
        f"TRUE:"
        f" {row['true_category']}"
    )

    print(
        f"FINE-TUNED:"
        f" {row['finetuned_prediction']}"
    )

    print(
        f"Top-1 score:"
        f" {row['finetuned_top1_score']:.6f}"
    )

    print(
        f"Top-2:"
        f" {row['finetuned_top2_category']}"
    )

    print(
        f"Top-2 score:"
        f" {row['finetuned_top2_score']:.6f}"
    )

    print(
        f"Margin:"
        f" {row['finetuned_margin']:.6f}"
    )

    print()
    print(
        f"TF-IDF:"
        f" {row['tfidf_prediction']}"
        f" | correct="
        f"{row['tfidf_correct']}"
    )

    print(
        f"Frozen BERTurk:"
        f" {row['frozen_prediction']}"
        f" | correct="
        f"{row['frozen_correct']}"
    )

    print()
    print(
        "Complaint:"
    )

    print(
        row["complaint_text"]
    )


print()
print("=" * 80)

print(
    f"Manual review CSV : "
    f"{FINETUNED_ERRORS_FILE}"
)

print(
    f"Error summary CSV : "
    f"{MODEL_ERROR_SUMMARY_FILE}"
)

print(
    f"Confusion pairs   : "
    f"{CONFUSION_PAIRS_FILE}"
)

print(
    f"Summary JSON      : "
    f"{SUMMARY_FILE}"
)

print()
print(
    "Post-test tuning  : NOT ALLOWED"
)

print()
print(
    "STATUS: "
    "FINAL_TEST_ERROR_ANALYSIS_PREPARED"
)