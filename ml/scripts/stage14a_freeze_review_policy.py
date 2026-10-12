import hashlib
import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

OPERATING_POINTS_FILE = Path(
    "ml/reports/model_evaluation/"
    "12d_validation_operating_points.csv"
)

CONFIG_DIR = Path(
    "ml/config"
)

REPORT_DIR = Path(
    "ml/reports/model_evaluation"
)

POLICY_FILE = (
    CONFIG_DIR
    / "review_policy_v1.json"
)

REPORT_FILE = (
    REPORT_DIR
    / "14a_review_policy_freeze_summary.json"
)


SELECTED_RULE = "TOP1_OR_MARGIN"
SELECTED_QUANTILE = 0.05


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path):

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


# ============================================================
# LOAD VALIDATION OPERATING POINTS
# ============================================================

if not OPERATING_POINTS_FILE.exists():

    raise FileNotFoundError(
        f"Missing file:\n"
        f"{OPERATING_POINTS_FILE}"
    )


source_hash = sha256_file(
    OPERATING_POINTS_FILE
)


df = pd.read_csv(
    OPERATING_POINTS_FILE
)


required_columns = {
    "rule",
    "source_quantile",
    "top1_threshold",
    "margin_threshold",
    "accepted_count",
    "review_count",
    "automatic_coverage",
    "review_rate",
    "accepted_accuracy",
}


missing = (
    required_columns
    - set(df.columns)
)

if missing:

    raise RuntimeError(
        "Missing columns:\n"
        + "\n".join(
            sorted(missing)
        )
    )


selected = df[
    (
        df["rule"]
        == SELECTED_RULE
    )
    &
    (
        (
            df["source_quantile"]
            - SELECTED_QUANTILE
        ).abs()
        < 1e-12
    )
]


if len(selected) != 1:

    raise RuntimeError(
        "Expected exactly one matching "
        "validation operating point."
    )


row = selected.iloc[0]


top1_threshold = float(
    row["top1_threshold"]
)

margin_threshold = float(
    row["margin_threshold"]
)

automatic_coverage = float(
    row["automatic_coverage"]
)

review_rate = float(
    row["review_rate"]
)

accepted_accuracy = float(
    row["accepted_accuracy"]
)


# ============================================================
# SANITY CHECKS
# ============================================================

if not (
    0.0
    < top1_threshold
    < 1.0
):

    raise RuntimeError(
        "Invalid top1 threshold."
    )


if not (
    0.0
    < margin_threshold
    < 1.0
):

    raise RuntimeError(
        "Invalid margin threshold."
    )


if not (
    0.04
    <= review_rate
    <= 0.07
):

    raise RuntimeError(
        "Unexpected validation review rate."
    )


if accepted_accuracy < 0.999:

    raise RuntimeError(
        "Selected validation operating "
        "point does not have expected "
        "accepted accuracy."
    )


# ============================================================
# POLICY
# ============================================================

policy = {
    "policy_id":
        "berturk_review_policy_v1",

    "model_family":
        "full_finetuned_berturk",

    "selection_basis":
        (
            "validation_lower_5_percent_"
            "operating_point"
        ),

    "selection_rationale":
        (
            "Use an operational manual-review "
            "budget of approximately five percent "
            "on the validation distribution. "
            "The threshold is derived only from "
            "precomputed validation predictions."
        ),

    "rule":
        "TOP1_OR_MARGIN",

    "top1_threshold":
        top1_threshold,

    "margin_threshold":
        margin_threshold,

    "decision_logic": {
        "NORMAL":
            (
                "top1_score >= top1_threshold "
                "AND score_margin >= "
                "margin_threshold"
            ),

        "REVIEW_LOW_CONFIDENCE":
            (
                "top1_score < "
                "top1_threshold"
            ),

        "REVIEW_CLOSE_TOP2":
            (
                "score_margin < "
                "margin_threshold"
            ),

        "INVALID_INPUT":
            (
                "Handled separately by "
                "application input validation."
            ),
    },

    "validation_behavior": {
        "automatic_coverage":
            automatic_coverage,

        "review_rate":
            review_rate,

        "accepted_accuracy":
            accepted_accuracy,
    },

    "score_interpretation":
        (
            "Softmax outputs are model scores "
            "and are not presented as guaranteed "
            "calibrated probabilities."
        ),

    "threshold_source":
        "validation_only",

    "standard_test_predictions_used":
        False,

    "challenge_predictions_used":
        False,

    "standard_test_had_already_been_opened":
        True,

    "post_test_model_tuning_allowed":
        False,

    "limitations": [
        (
            "Validation contained zero "
            "classification errors, so no "
            "empirical correct-versus-incorrect "
            "error boundary could be estimated."
        ),
        (
            "High-confidence errors may not "
            "be detected by confidence-based "
            "review rules."
        ),
        (
            "OOD inputs may still receive "
            "high softmax scores because the "
            "classifier is closed-set."
        ),
    ],

    "source_file":
        str(
            OPERATING_POINTS_FILE
        ),

    "source_sha256":
        source_hash,
}


# ============================================================
# SAVE
# ============================================================

CONFIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with POLICY_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        policy,
        f,
        ensure_ascii=False,
        indent=2,
    )


summary = {
    "status":
        "REVIEW_POLICY_V1_FROZEN",

    "policy_file":
        str(POLICY_FILE),

    "selected_rule":
        SELECTED_RULE,

    "source_quantile":
        SELECTED_QUANTILE,

    "top1_threshold":
        top1_threshold,

    "margin_threshold":
        margin_threshold,

    "validation_automatic_coverage":
        automatic_coverage,

    "validation_review_rate":
        review_rate,

    "validation_accepted_accuracy":
        accepted_accuracy,

    "threshold_source":
        "validation_only",

    "test_used_for_threshold":
        False,

    "challenge_used_for_threshold":
        False,

    "post_test_model_tuning_allowed":
        False,
}


with REPORT_FILE.open(
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
    "STAGE 14A - REVIEW POLICY FREEZE"
)
print("=" * 80)

print(
    f"Rule                : "
    f"{SELECTED_RULE}"
)

print(
    f"Validation quantile : "
    f"{SELECTED_QUANTILE}"
)

print(
    f"Top-1 threshold     : "
    f"{top1_threshold:.9f}"
)

print(
    f"Margin threshold    : "
    f"{margin_threshold:.9f}"
)

print(
    f"Automatic coverage  : "
    f"{automatic_coverage:.3%}"
)

print(
    f"Manual review rate  : "
    f"{review_rate:.3%}"
)

print(
    f"Accepted accuracy   : "
    f"{accepted_accuracy:.3%}"
)

print()
print(
    "Threshold source    : VALIDATION ONLY"
)

print(
    "Standard TEST used  : False"
)

print(
    "Challenge used      : False"
)

print(
    "Post-test tuning    : NOT ALLOWED"
)

print()
print(
    f"Policy              : "
    f"{POLICY_FILE}"
)

print(
    f"Summary             : "
    f"{REPORT_FILE}"
)

print()
print(
    "STATUS: REVIEW_POLICY_V1_FROZEN"
)