import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# INPUT / OUTPUT
# ============================================================

PREDICTIONS_FILE = Path(
    "ml/models/berturk/final_model/"
    "validation_predictions.csv"
)

REPORT_DIR = Path(
    "ml/reports/model_evaluation"
)

SUMMARY_FILE = REPORT_DIR / (
    "12d_validation_confidence_summary.json"
)

OPERATING_POINTS_FILE = REPORT_DIR / (
    "12d_validation_operating_points.csv"
)

LOWEST_CONFIDENCE_FILE = REPORT_DIR / (
    "12d_lowest_confidence_examples.csv"
)

PER_CLASS_FILE = REPORT_DIR / (
    "12d_validation_confidence_per_class.csv"
)


EXPECTED_COUNT = 250


# ============================================================
# LOAD
# ============================================================

if not PREDICTIONS_FILE.exists():

    raise FileNotFoundError(
        f"Dosya bulunamadı:\n"
        f"{PREDICTIONS_FILE}"
    )


df = pd.read_csv(
    PREDICTIONS_FILE
)


required_columns = {
    "sample_id",
    "complaint_text",
    "true_category",
    "predicted_category",
    "correct",
    "top1_score",
    "top2_score",
    "score_margin",
}


missing = (
    required_columns
    - set(df.columns)
)

if missing:

    raise RuntimeError(
        "Eksik kolonlar:\n"
        + "\n".join(
            sorted(missing)
        )
    )


if len(df) != EXPECTED_COUNT:

    raise RuntimeError(
        f"Expected {EXPECTED_COUNT}, "
        f"found {len(df)}"
    )


# bool/string compatibility
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
        "correct kolonu parse edilemedi."
    )


correct_count = int(
    df["correct"].sum()
)

incorrect_count = int(
    (~df["correct"]).sum()
)


# ============================================================
# BASIC DISTRIBUTION
# ============================================================

def distribution_stats(series):

    values = np.asarray(
        series,
        dtype=float,
    )

    return {
        "min":
            float(
                np.min(values)
            ),

        "p01":
            float(
                np.quantile(
                    values,
                    0.01,
                )
            ),

        "p025":
            float(
                np.quantile(
                    values,
                    0.025,
                )
            ),

        "p05":
            float(
                np.quantile(
                    values,
                    0.05,
                )
            ),

        "p10":
            float(
                np.quantile(
                    values,
                    0.10,
                )
            ),

        "median":
            float(
                np.median(values)
            ),

        "mean":
            float(
                np.mean(values)
            ),

        "p90":
            float(
                np.quantile(
                    values,
                    0.90,
                )
            ),

        "max":
            float(
                np.max(values)
            ),
    }


top1_stats = distribution_stats(
    df["top1_score"]
)

top2_stats = distribution_stats(
    df["top2_score"]
)

margin_stats = distribution_stats(
    df["score_margin"]
)


# ============================================================
# PER-CLASS CONFIDENCE
# ============================================================

per_class_rows = []


for category, group in (
    df.groupby(
        "true_category"
    )
):

    per_class_rows.append({
        "category":
            category,

        "count":
            len(group),

        "accuracy":
            float(
                group["correct"].mean()
            ),

        "top1_min":
            float(
                group[
                    "top1_score"
                ].min()
            ),

        "top1_mean":
            float(
                group[
                    "top1_score"
                ].mean()
            ),

        "top1_median":
            float(
                group[
                    "top1_score"
                ].median()
            ),

        "margin_min":
            float(
                group[
                    "score_margin"
                ].min()
            ),

        "margin_mean":
            float(
                group[
                    "score_margin"
                ].mean()
            ),

        "margin_median":
            float(
                group[
                    "score_margin"
                ].median()
            ),
    })


per_class_df = pd.DataFrame(
    per_class_rows
).sort_values(
    "category"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

per_class_df.to_csv(
    PER_CLASS_FILE,
    index=False,
)


# ============================================================
# DATA-DERIVED OPERATING POINTS
#
# q means:
# use the q-th lower quantile of validation
# confidence distribution as a candidate threshold.
#
# These are CANDIDATES, not yet final policy.
# ============================================================

quantiles = [
    0.01,
    0.025,
    0.05,
    0.10,
]


operating_rows = []


def evaluate_rule(
    rule_name,
    review_mask,
    top1_threshold=None,
    margin_threshold=None,
    source_quantile=None,
):

    review_mask = np.asarray(
        review_mask,
        dtype=bool,
    )

    accepted_mask = (
        ~review_mask
    )

    review_count = int(
        review_mask.sum()
    )

    accepted_count = int(
        accepted_mask.sum()
    )

    review_rate = (
        review_count
        / len(df)
    )

    coverage = (
        accepted_count
        / len(df)
    )

    if accepted_count > 0:

        accepted_accuracy = float(
            df.loc[
                accepted_mask,
                "correct",
            ].mean()
        )

    else:

        accepted_accuracy = None


    if review_count > 0:

        reviewed_accuracy = float(
            df.loc[
                review_mask,
                "correct",
            ].mean()
        )

    else:

        reviewed_accuracy = None


    operating_rows.append({
        "rule":
            rule_name,

        "source_quantile":
            source_quantile,

        "top1_threshold":
            top1_threshold,

        "margin_threshold":
            margin_threshold,

        "accepted_count":
            accepted_count,

        "review_count":
            review_count,

        "automatic_coverage":
            coverage,

        "review_rate":
            review_rate,

        "accepted_accuracy":
            accepted_accuracy,

        "reviewed_accuracy":
            reviewed_accuracy,
    })


for q in quantiles:

    top1_threshold = float(
        df[
            "top1_score"
        ].quantile(q)
    )

    margin_threshold = float(
        df[
            "score_margin"
        ].quantile(q)
    )


    # ----------------------------------------
    # Top-1-only candidate
    # ----------------------------------------

    evaluate_rule(
        rule_name="TOP1_ONLY",
        review_mask=(
            df["top1_score"]
            < top1_threshold
        ),
        top1_threshold=(
            top1_threshold
        ),
        margin_threshold=None,
        source_quantile=q,
    )


    # ----------------------------------------
    # Margin-only candidate
    # ----------------------------------------

    evaluate_rule(
        rule_name="MARGIN_ONLY",
        review_mask=(
            df["score_margin"]
            < margin_threshold
        ),
        top1_threshold=None,
        margin_threshold=(
            margin_threshold
        ),
        source_quantile=q,
    )


    # ----------------------------------------
    # Combined conservative candidate
    #
    # Review when:
    # low top-1 OR close top-2 margin
    # ----------------------------------------

    evaluate_rule(
        rule_name="TOP1_OR_MARGIN",
        review_mask=(
            (
                df["top1_score"]
                < top1_threshold
            )
            |
            (
                df["score_margin"]
                < margin_threshold
            )
        ),
        top1_threshold=(
            top1_threshold
        ),
        margin_threshold=(
            margin_threshold
        ),
        source_quantile=q,
    )


operating_df = pd.DataFrame(
    operating_rows
)

operating_df.to_csv(
    OPERATING_POINTS_FILE,
    index=False,
)


# ============================================================
# LOWEST CONFIDENCE EXAMPLES
# ============================================================

lowest_df = (
    df[
        [
            "sample_id",
            "complaint_text",
            "true_category",
            "predicted_category",
            "correct",
            "top1_score",
            "top2_score",
            "score_margin",
        ]
    ]
    .sort_values(
        [
            "top1_score",
            "score_margin",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .head(20)
)


lowest_df.to_csv(
    LOWEST_CONFIDENCE_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "status":
        (
            "VALIDATION_CONFIDENCE_"
            "ANALYSIS_COMPLETE"
        ),

    "validation_count":
        len(df),

    "correct_count":
        correct_count,

    "incorrect_count":
        incorrect_count,

    "validation_accuracy":
        float(
            df["correct"].mean()
        ),

    "top1_distribution":
        top1_stats,

    "top2_distribution":
        top2_stats,

    "margin_distribution":
        margin_stats,

    "operating_point_quantiles":
        quantiles,

    "threshold_finalized":
        False,

    "reason_threshold_not_finalized":
        (
            "All 250 validation samples "
            "were classified correctly. "
            "Validation therefore provides "
            "confidence-distribution and "
            "coverage information, but no "
            "observed error boundary from "
            "which a final error-separating "
            "threshold can be estimated."
        ),

    "policy_note":
        (
            "Softmax scores are treated as "
            "model scores, not guaranteed "
            "calibrated probabilities. "
            "Candidate thresholds are derived "
            "from validation quantiles rather "
            "than arbitrary constants."
        ),

    "frozen_test_accessed":
        False,
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

print("=" * 78)
print(
    "STAGE 12D - VALIDATION "
    "CONFIDENCE ANALYSIS"
)
print("=" * 78)

print(
    f"Validation samples : "
    f"{len(df)}"
)

print(
    f"Correct            : "
    f"{correct_count}"
)

print(
    f"Incorrect          : "
    f"{incorrect_count}"
)


print("\nTOP-1 SCORE")

for key, value in (
    top1_stats.items()
):

    print(
        f"  {key:<8}: "
        f"{value:.6f}"
    )


print("\nTOP1-TOP2 MARGIN")

for key, value in (
    margin_stats.items()
):

    print(
        f"  {key:<8}: "
        f"{value:.6f}"
    )


print("\nOPERATING POINT CANDIDATES")

display_columns = [
    "rule",
    "source_quantile",
    "top1_threshold",
    "margin_threshold",
    "automatic_coverage",
    "review_rate",
    "accepted_accuracy",
]


for _, row in (
    operating_df.iterrows()
):

    print(
        f"{row['rule']:<16} "
        f"q={row['source_quantile']:<5} "
        f"coverage="
        f"{row['automatic_coverage']:.3f} "
        f"review="
        f"{row['review_rate']:.3f} "
        f"accepted_acc="
        f"{row['accepted_accuracy']:.3f}"
    )


print("\nLowest-confidence examples:")

for _, row in (
    lowest_df.head(10)
    .iterrows()
):

    print(
        f"  {row['sample_id']} | "
        f"{row['true_category']} | "
        f"top1={row['top1_score']:.6f} | "
        f"margin={row['score_margin']:.6f}"
    )


print("\nOutputs:")

print(
    f"  Summary          : "
    f"{SUMMARY_FILE}"
)

print(
    f"  Operating points : "
    f"{OPERATING_POINTS_FILE}"
)

print(
    f"  Per class        : "
    f"{PER_CLASS_FILE}"
)

print(
    f"  Lowest examples  : "
    f"{LOWEST_CONFIDENCE_FILE}"
)

print()
print(
    "Threshold finalized : False"
)

print(
    "Frozen TEST accessed: False"
)

print()
print(
    "STATUS: "
    "VALIDATION_CONFIDENCE_ANALYSIS_COMPLETE"
)