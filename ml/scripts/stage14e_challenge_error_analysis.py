import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PREDICTIONS_FILE = Path(
    "ml/reports/challenge/"
    "14d_challenge_predictions.csv"
)

SUMMARY_FILE = Path(
    "ml/reports/challenge/"
    "14d_challenge_evaluation_summary.json"
)

OUTPUT_DIR = Path(
    "ml/reports/challenge/"
    "error_analysis"
)

IN_SCOPE_ERRORS_FILE = (
    OUTPUT_DIR
    / "14e_in_scope_errors_for_manual_review.csv"
)

OOD_AUTOACCEPT_FILE = (
    OUTPUT_DIR
    / "14e_ood_autoaccepted_for_manual_review.csv"
)

CATEGORY_SUMMARY_FILE = (
    OUTPUT_DIR
    / "14e_in_scope_by_category.csv"
)

REVIEW_BEHAVIOR_FILE = (
    OUTPUT_DIR
    / "14e_review_behavior_summary.csv"
)

ANALYSIS_SUMMARY_FILE = (
    OUTPUT_DIR
    / "14e_challenge_error_analysis_summary.json"
)


# ============================================================
# LOAD
# ============================================================

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Missing predictions file:\n"
        f"{PREDICTIONS_FILE}"
    )

if not SUMMARY_FILE.exists():
    raise FileNotFoundError(
        f"Missing summary file:\n"
        f"{SUMMARY_FILE}"
    )


df = pd.read_csv(
    PREDICTIONS_FILE
)

with SUMMARY_FILE.open(
    "r",
    encoding="utf-8",
) as f:
    stage14d_summary = json.load(f)


required_columns = {
    "challenge_id",
    "challenge_group",
    "complaint_text",
    "gold_category",
    "ood_family",
    "challenge_type",
    "predicted_category",
    "correct",
    "top1_score",
    "top2_category",
    "top2_score",
    "score_margin",
    "review_required",
    "review_status",
}

missing = required_columns - set(df.columns)

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(
            sorted(missing)
        )
    )


# ============================================================
# NORMALIZE BOOLS
# ============================================================

def normalize_bool(series):

    if series.dtype == object:
        return (
            series.astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False,
            })
        )

    return series.astype(bool)


df["review_required"] = normalize_bool(
    df["review_required"]
)

if df["review_required"].isna().any():
    raise RuntimeError(
        "review_required could not "
        "be parsed."
    )


# "correct" is null for OOD, so only normalize in-scope.
in_scope_mask = (
    df["challenge_group"]
    == "IN_SCOPE"
)

ood_mask = (
    df["challenge_group"]
    == "OOD"
)

in_scope_df = df[
    in_scope_mask
].copy()

ood_df = df[
    ood_mask
].copy()


in_scope_df["correct"] = (
    normalize_bool(
        in_scope_df[
            "correct"
        ]
    )
)

if in_scope_df["correct"].isna().any():
    raise RuntimeError(
        "IN_SCOPE correct column "
        "could not be parsed."
    )


if len(in_scope_df) != 100:
    raise RuntimeError(
        f"Expected 100 IN_SCOPE, "
        f"found {len(in_scope_df)}"
    )

if len(ood_df) != 50:
    raise RuntimeError(
        f"Expected 50 OOD, "
        f"found {len(ood_df)}"
    )


# ============================================================
# IN-SCOPE ERRORS
# ============================================================

errors_df = (
    in_scope_df[
        ~in_scope_df[
            "correct"
        ]
    ]
    .copy()
)


errors_df[
    "review_caught_error"
] = (
    errors_df[
        "review_required"
    ]
)


errors_df[
    "manual_error_type"
] = ""

errors_df[
    "manual_notes"
] = ""

errors_df[
    "gold_label_suspected"
] = False


# ============================================================
# OOD AUTO-ACCEPTED
# ============================================================

ood_autoaccepted_df = (
    ood_df[
        ~ood_df[
            "review_required"
        ]
    ]
    .copy()
)


ood_autoaccepted_df[
    "manual_notes"
] = ""

ood_autoaccepted_df[
    "ood_status_suspected"
] = False


# ============================================================
# CATEGORY-LEVEL IN-SCOPE SUMMARY
# ============================================================

category_rows = []

for category, group in (
    in_scope_df.groupby(
        "gold_category"
    )
):

    total = len(group)

    correct_count = int(
        group[
            "correct"
        ].sum()
    )

    error_count = (
        total
        - correct_count
    )

    review_count = int(
        group[
            "review_required"
        ].sum()
    )

    accepted = group[
        ~group[
            "review_required"
        ]
    ]

    if len(accepted) > 0:
        accepted_accuracy = float(
            accepted[
                "correct"
            ].mean()
        )
    else:
        accepted_accuracy = None

    if error_count > 0:
        caught_errors = int(
            group.loc[
                ~group[
                    "correct"
                ],
                "review_required",
            ].sum()
        )

        error_capture_rate = (
            caught_errors
            / error_count
        )
    else:
        caught_errors = 0
        error_capture_rate = None

    category_rows.append({
        "gold_category":
            category,

        "count":
            total,

        "accuracy":
            float(
                group[
                    "correct"
                ].mean()
            ),

        "error_count":
            error_count,

        "review_count":
            review_count,

        "review_rate":
            review_count / total,

        "accepted_count":
            len(accepted),

        "accepted_accuracy":
            accepted_accuracy,

        "caught_errors":
            caught_errors,

        "error_capture_rate":
            error_capture_rate,

        "mean_top1_score":
            float(
                group[
                    "top1_score"
                ].mean()
            ),

        "mean_margin":
            float(
                group[
                    "score_margin"
                ].mean()
            ),
    })


category_df = pd.DataFrame(
    category_rows
).sort_values(
    "gold_category"
)


# ============================================================
# REVIEW BEHAVIOR SUMMARY
# ============================================================

in_scope_reviewed = (
    in_scope_df[
        in_scope_df[
            "review_required"
        ]
    ]
)

in_scope_accepted = (
    in_scope_df[
        ~in_scope_df[
            "review_required"
        ]
    ]
)


caught_errors = int(
    errors_df[
        "review_required"
    ].sum()
)

missed_errors = (
    len(errors_df)
    - caught_errors
)

correct_reviewed = int(
    in_scope_reviewed[
        "correct"
    ].sum()
)

incorrect_reviewed = (
    len(in_scope_reviewed)
    - correct_reviewed
)

incorrect_autoaccepted = int(
    (
        ~in_scope_accepted[
            "correct"
        ]
    ).sum()
)


review_behavior_df = pd.DataFrame([
    {
        "group":
            "IN_SCOPE",

        "total":
            len(in_scope_df),

        "reviewed":
            len(in_scope_reviewed),

        "auto_accepted":
            len(in_scope_accepted),

        "errors_total":
            len(errors_df),

        "errors_caught_by_review":
            caught_errors,

        "errors_missed_by_review":
            missed_errors,

        "correct_samples_sent_to_review":
            correct_reviewed,

        "incorrect_samples_sent_to_review":
            incorrect_reviewed,

        "incorrect_autoaccepted":
            incorrect_autoaccepted,
    },
    {
        "group":
            "OOD",

        "total":
            len(ood_df),

        "reviewed":
            int(
                ood_df[
                    "review_required"
                ].sum()
            ),

        "auto_accepted":
            len(
                ood_autoaccepted_df
            ),

        "errors_total":
            None,

        "errors_caught_by_review":
            None,

        "errors_missed_by_review":
            None,

        "correct_samples_sent_to_review":
            None,

        "incorrect_samples_sent_to_review":
            None,

        "incorrect_autoaccepted":
            None,
    },
])


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


errors_df.to_csv(
    IN_SCOPE_ERRORS_FILE,
    index=False,
)


ood_autoaccepted_df.to_csv(
    OOD_AUTOACCEPT_FILE,
    index=False,
)


category_df.to_csv(
    CATEGORY_SUMMARY_FILE,
    index=False,
)


review_behavior_df.to_csv(
    REVIEW_BEHAVIOR_FILE,
    index=False,
)


summary = {
    "status":
        "CHALLENGE_ERROR_ANALYSIS_PREPARED",

    "in_scope": {
        "count":
            len(in_scope_df),

        "errors":
            len(errors_df),

        "reviewed":
            len(in_scope_reviewed),

        "auto_accepted":
            len(in_scope_accepted),

        "errors_caught_by_review":
            caught_errors,

        "errors_missed_by_review":
            missed_errors,

        "correct_samples_sent_to_review":
            correct_reviewed,

        "incorrect_autoaccepted":
            incorrect_autoaccepted,
    },

    "ood": {
        "count":
            len(ood_df),

        "reviewed":
            int(
                ood_df[
                    "review_required"
                ].sum()
            ),

        "auto_accepted":
            len(
                ood_autoaccepted_df
            ),
    },

    "model_tuning_allowed":
        False,

    "threshold_tuning_allowed":
        False,

    "note":
        (
            "This stage prepares manual "
            "analysis of real challenge-set "
            "errors and high-confidence OOD "
            "failures. It does not modify the "
            "model or the frozen review policy."
        ),
}


with ANALYSIS_SUMMARY_FILE.open(
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
# TERMINAL
# ============================================================

print("=" * 80)
print(
    "STAGE 14E - CHALLENGE "
    "ERROR / REVIEW ANALYSIS"
)
print("=" * 80)


print()
print(
    "IN-SCOPE REVIEW BEHAVIOR"
)

print(
    f"Total               : "
    f"{len(in_scope_df)}"
)

print(
    f"Errors              : "
    f"{len(errors_df)}"
)

print(
    f"Sent to review      : "
    f"{len(in_scope_reviewed)}"
)

print(
    f"Auto accepted       : "
    f"{len(in_scope_accepted)}"
)

print(
    f"Errors caught       : "
    f"{caught_errors}/"
    f"{len(errors_df)}"
)

print(
    f"Errors missed       : "
    f"{missed_errors}"
)

print(
    f"Correct sent review : "
    f"{correct_reviewed}"
)

print(
    f"Wrong auto-accepted : "
    f"{incorrect_autoaccepted}"
)


print()
print(
    "IN-SCOPE ERRORS"
)


for _, row in (
    errors_df.sort_values(
        [
            "review_required",
            "top1_score",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .iterrows()
):

    print()
    print("-" * 80)

    print(
        f"ID:"
        f" {row['challenge_id']}"
    )

    print(
        f"Type:"
        f" {row['challenge_type']}"
    )

    print(
        f"TRUE:"
        f" {row['gold_category']}"
    )

    print(
        f"PRED:"
        f" {row['predicted_category']}"
    )

    print(
        f"Top-1:"
        f" {row['top1_score']:.6f}"
    )

    print(
        f"Top-2:"
        f" {row['top2_category']}"
        f" ({row['top2_score']:.6f})"
    )

    print(
        f"Margin:"
        f" {row['score_margin']:.6f}"
    )

    print(
        f"Review:"
        f" {row['review_required']}"
        f" | {row['review_status']}"
    )

    print(
        "Text:"
    )

    print(
        row[
            "complaint_text"
        ]
    )


print()
print("=" * 80)

print(
    "HIGH-CONFIDENCE OOD "
    "AUTO-ACCEPTED"
)

print("=" * 80)

print(
    f"Count               : "
    f"{len(ood_autoaccepted_df)}"
)


for _, row in (
    ood_autoaccepted_df
    .sort_values(
        "top1_score",
        ascending=False,
    )
    .iterrows()
):

    print()
    print("-" * 80)

    print(
        f"ID:"
        f" {row['challenge_id']}"
    )

    print(
        f"OOD family:"
        f" {row['ood_family']}"
    )

    print(
        f"Forced prediction:"
        f" {row['predicted_category']}"
    )

    print(
        f"Top-1:"
        f" {row['top1_score']:.6f}"
    )

    print(
        f"Top-2:"
        f" {row['top2_category']}"
        f" ({row['top2_score']:.6f})"
    )

    print(
        f"Margin:"
        f" {row['score_margin']:.6f}"
    )

    print(
        "Text:"
    )

    print(
        row[
            "complaint_text"
        ]
    )


print()
print("=" * 80)
print(
    "CATEGORY SUMMARY"
)
print("=" * 80)


for _, row in (
    category_df.iterrows()
):

    capture = (
        "N/A"
        if pd.isna(
            row[
                "error_capture_rate"
            ]
        )
        else
        f"{row['error_capture_rate']:.3f}"
    )

    accepted_acc = (
        "N/A"
        if pd.isna(
            row[
                "accepted_accuracy"
            ]
        )
        else
        f"{row['accepted_accuracy']:.3f}"
    )

    print(
        f"{row['gold_category']:<24} "
        f"acc="
        f"{row['accuracy']:.3f} | "
        f"errors="
        f"{int(row['error_count'])} | "
        f"review="
        f"{row['review_rate']:.3f} | "
        f"accepted_acc="
        f"{accepted_acc} | "
        f"error_capture="
        f"{capture}"
    )


print()
print(
    f"IN-scope errors CSV : "
    f"{IN_SCOPE_ERRORS_FILE}"
)

print(
    f"OOD autoaccept CSV  : "
    f"{OOD_AUTOACCEPT_FILE}"
)

print(
    f"Category summary    : "
    f"{CATEGORY_SUMMARY_FILE}"
)

print(
    f"Review summary      : "
    f"{REVIEW_BEHAVIOR_FILE}"
)

print(
    f"Summary JSON        : "
    f"{ANALYSIS_SUMMARY_FILE}"
)

print()
print(
    "Model tuning        : NOT ALLOWED"
)

print(
    "Threshold tuning    : NOT ALLOWED"
)

print()
print(
    "STATUS: "
    "CHALLENGE_ERROR_ANALYSIS_PREPARED"
)
