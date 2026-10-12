import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# CONFIG
# ============================================================

CHALLENGE_FILE = Path(
    "data/challenge/challenge_set_final_v1.jsonl"
)

REVIEW_POLICY_FILE = Path(
    "ml/config/review_policy_v1.json"
)

FINETUNED_MODEL_DIR = Path(
    "ml/models/berturk/final_model"
)

FINETUNED_MODEL_WEIGHTS = (
    FINETUNED_MODEL_DIR
    / "model.safetensors"
)

REPORT_DIR = Path(
    "ml/reports/challenge"
)

EXPECTED_CHALLENGE_SHA256 = (
    "8c70ad783976fdb58597b5feddeffbb8"
    "e23c676b149e5119b005605888dcd7d8"
)

EXPECTED_FINETUNED_MODEL_SHA256 = (
    "af3aaf2d03972e0b07f5858971b9114b"
    "efea21b7127e390d088e3c85e7bd1bb4"
)

MAX_LENGTH = 160
BATCH_SIZE = 8

LABELS = [
    "TEMIZLIK_ATIK",
    "YOL_KALDIRIM",
    "PARK_YESIL_ALAN",
    "KENT_DUZENI_DENETIM",
    "HAYVAN_VETERINER",
]

LABEL_TO_ID = {
    label: i
    for i, label in enumerate(LABELS)
}

ID_TO_LABEL = {
    i: label
    for label, i in LABEL_TO_ID.items()
}


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


def load_jsonl(path: Path):

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as e:

                raise RuntimeError(
                    f"Invalid JSON at line "
                    f"{line_no}: {e}"
                )

    return records


def find_model_root(path: Path):

    direct_config = (
        path / "config.json"
    )

    if direct_config.exists():

        return path

    candidates = list(
        path.rglob(
            "config.json"
        )
    )

    if len(candidates) != 1:

        raise RuntimeError(
            "Fine-tuned model root "
            "could not be determined."
        )

    return candidates[0].parent


def get_review_status(
    top1_score,
    margin,
    top1_threshold,
    margin_threshold,
):

    low_confidence = (
        top1_score
        < top1_threshold
    )

    close_top2 = (
        margin
        < margin_threshold
    )

    if (
        low_confidence
        and close_top2
    ):

        return (
            "REVIEW_LOW_CONFIDENCE_"
            "AND_CLOSE_TOP2"
        )

    if low_confidence:

        return (
            "REVIEW_LOW_CONFIDENCE"
        )

    if close_top2:

        return (
            "REVIEW_CLOSE_TOP2"
        )

    return "NORMAL"


def classification_metrics(
    y_true,
    y_pred,
    probabilities,
):

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    y_true,
                    probabilities,
                    labels=list(
                        range(
                            len(LABELS)
                        )
                    ),
                )
            ),
    }


# ============================================================
# VERIFY INPUTS
# ============================================================

required_paths = [
    CHALLENGE_FILE,
    REVIEW_POLICY_FILE,
    FINETUNED_MODEL_DIR,
    FINETUNED_MODEL_WEIGHTS,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required path:\n"
            f"{path}"
        )


challenge_hash = sha256_file(
    CHALLENGE_FILE
)

finetuned_model_hash = (
    sha256_file(
        FINETUNED_MODEL_WEIGHTS
    )
)


if (
    challenge_hash
    != EXPECTED_CHALLENGE_SHA256
):

    raise RuntimeError(
        "Challenge SHA256 mismatch.\n"
        f"Expected: "
        f"{EXPECTED_CHALLENGE_SHA256}\n"
        f"Found   : "
        f"{challenge_hash}"
    )


if (
    finetuned_model_hash
    != EXPECTED_FINETUNED_MODEL_SHA256
):

    raise RuntimeError(
        "Fine-tuned model "
        "SHA256 mismatch.\n"
        f"Expected: "
        f"{EXPECTED_FINETUNED_MODEL_SHA256}\n"
        f"Found   : "
        f"{finetuned_model_hash}"
    )


with REVIEW_POLICY_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    review_policy = json.load(
        f
    )


if (
    review_policy.get(
        "policy_id"
    )
    != "berturk_review_policy_v1"
):

    raise RuntimeError(
        "Unexpected review policy."
    )


top1_threshold = float(
    review_policy[
        "top1_threshold"
    ]
)

margin_threshold = float(
    review_policy[
        "margin_threshold"
    ]
)


# ============================================================
# LOAD CHALLENGE SET
# ============================================================

records = load_jsonl(
    CHALLENGE_FILE
)


if len(records) != 150:

    raise RuntimeError(
        f"Expected 150 challenge "
        f"records, found {len(records)}"
    )


in_scope_records = [
    r
    for r in records
    if r.get(
        "challenge_group"
    ) == "IN_SCOPE"
]

ood_records = [
    r
    for r in records
    if r.get(
        "challenge_group"
    ) == "OOD"
]


if len(in_scope_records) != 100:

    raise RuntimeError(
        "Expected 100 IN_SCOPE records."
    )


if len(ood_records) != 50:

    raise RuntimeError(
        "Expected 50 OOD records."
    )


for record in in_scope_records:

    if (
        record.get(
            "gold_category"
        )
        not in LABEL_TO_ID
    ):

        raise RuntimeError(
            "Invalid in-scope gold label."
        )


for record in ood_records:

    if (
        record.get(
            "gold_category"
        )
        is not None
    ):

        raise RuntimeError(
            "OOD gold_category must "
            "be null."
        )


texts = [
    r["complaint_text"]
    for r in records
]


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model_root = find_model_root(
    FINETUNED_MODEL_DIR
)


print("=" * 80)
print(
    "STAGE 14D - FROZEN CHALLENGE "
    "EVALUATION"
)
print("=" * 80)

print(
    f"Challenge records   : "
    f"{len(records)}"
)

print(
    f"IN_SCOPE            : "
    f"{len(in_scope_records)}"
)

print(
    f"OOD                 : "
    f"{len(ood_records)}"
)

print(
    f"Challenge SHA256    : "
    f"{challenge_hash}"
)

print(
    f"Review top1 thr.    : "
    f"{top1_threshold:.9f}"
)

print(
    f"Review margin thr.  : "
    f"{margin_threshold:.9f}"
)

print(
    f"Device              : "
    f"{device}"
)

if device.type == "cuda":

    print(
        f"GPU                 : "
        f"{torch.cuda.get_device_name(0)}"
    )

print()
print(
    "Challenge is evaluation-only."
)

print(
    "No model/threshold tuning "
    "is permitted."
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        model_root
    )
)


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        model_root
    )
)


model.to(
    device
)

model.eval()


# ============================================================
# INFERENCE
# ============================================================

all_logits = []

start = time.perf_counter()


with torch.inference_mode():

    for batch_start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):

        batch_texts = texts[
            batch_start:
            batch_start
            + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key:
                value.to(
                    device
                )
            for key, value
            in encoded.items()
        }

        outputs = model(
            **encoded
        )

        all_logits.append(
            outputs.logits
            .detach()
            .cpu()
        )


elapsed_seconds = (
    time.perf_counter()
    - start
)


logits = torch.cat(
    all_logits,
    dim=0,
)


probabilities = (
    torch.softmax(
        logits,
        dim=-1,
    )
    .numpy()
)


predicted_ids = np.argmax(
    probabilities,
    axis=1,
).astype(
    np.int64
)


# ============================================================
# BUILD PREDICTION TABLE
# ============================================================

prediction_rows = []


for i, record in enumerate(
    records
):

    probs = probabilities[i]

    order = np.argsort(
        probs
    )[::-1]

    top1_id = int(
        order[0]
    )

    top2_id = int(
        order[1]
    )

    top1_score = float(
        probs[
            top1_id
        ]
    )

    top2_score = float(
        probs[
            top2_id
        ]
    )

    margin = (
        top1_score
        - top2_score
    )

    review_status = (
        get_review_status(
            top1_score,
            margin,
            top1_threshold,
            margin_threshold,
        )
    )

    review_required = (
        review_status != "NORMAL"
    )


    if (
        record[
            "challenge_group"
        ]
        == "IN_SCOPE"
    ):

        correct = (
            ID_TO_LABEL[
                top1_id
            ]
            == record[
                "gold_category"
            ]
        )

    else:

        correct = None


    row = {
        "challenge_id":
            record[
                "challenge_id"
            ],

        "challenge_group":
            record[
                "challenge_group"
            ],

        "complaint_text":
            record[
                "complaint_text"
            ],

        "gold_category":
            record.get(
                "gold_category"
            ),

        "ood_family":
            record.get(
                "ood_family"
            ),

        "challenge_type":
            record.get(
                "challenge_type"
            ),

        "predicted_category":
            ID_TO_LABEL[
                top1_id
            ],

        "correct":
            correct,

        "top1_score":
            top1_score,

        "top2_category":
            ID_TO_LABEL[
                top2_id
            ],

        "top2_score":
            top2_score,

        "score_margin":
            margin,

        "review_required":
            review_required,

        "review_status":
            review_status,
    }


    for j, label in enumerate(
        LABELS
    ):

        row[
            f"prob_{label}"
        ] = float(
            probs[j]
        )


    prediction_rows.append(
        row
    )


predictions_df = pd.DataFrame(
    prediction_rows
)


# ============================================================
# IN-SCOPE METRICS
# ============================================================

in_scope_df = (
    predictions_df[
        predictions_df[
            "challenge_group"
        ]
        == "IN_SCOPE"
    ]
    .copy()
)


y_true = np.asarray(
    [
        LABEL_TO_ID[x]
        for x in in_scope_df[
            "gold_category"
        ]
    ],
    dtype=np.int64,
)


y_pred = np.asarray(
    [
        LABEL_TO_ID[x]
        for x in in_scope_df[
            "predicted_category"
        ]
    ],
    dtype=np.int64,
)


in_scope_indices = (
    in_scope_df.index.to_numpy()
)


in_scope_probabilities = (
    probabilities[
        in_scope_indices
    ]
)


in_scope_metrics = (
    classification_metrics(
        y_true,
        y_pred,
        in_scope_probabilities,
    )
)


classification_report_dict = (
    classification_report(
        y_true,
        y_pred,
        labels=list(
            range(
                len(LABELS)
            )
        ),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
)


confusion = confusion_matrix(
    y_true,
    y_pred,
    labels=list(
        range(
            len(LABELS)
        )
    ),
)


in_scope_errors = (
    ~in_scope_df[
        "correct"
    ].astype(bool)
)


in_scope_error_count = int(
    in_scope_errors.sum()
)


in_scope_review_count = int(
    in_scope_df[
        "review_required"
    ].sum()
)


in_scope_review_rate = (
    in_scope_review_count
    / len(in_scope_df)
)


accepted_in_scope = (
    in_scope_df[
        ~in_scope_df[
            "review_required"
        ]
    ]
)


if len(
    accepted_in_scope
) > 0:

    accepted_accuracy = float(
        accepted_in_scope[
            "correct"
        ]
        .astype(bool)
        .mean()
    )

else:

    accepted_accuracy = None


if (
    in_scope_error_count
    > 0
):

    flagged_errors = int(
        in_scope_df.loc[
            in_scope_errors,
            "review_required",
        ].sum()
    )

    error_capture_rate = (
        flagged_errors
        / in_scope_error_count
    )

else:

    flagged_errors = 0
    error_capture_rate = None


# ============================================================
# CHALLENGE TYPE METRICS
# ============================================================

challenge_type_rows = []


for (
    challenge_type,
    group,
) in in_scope_df.groupby(
    "challenge_type"
):

    group_true = np.asarray(
        [
            LABEL_TO_ID[x]
            for x in group[
                "gold_category"
            ]
        ],
        dtype=np.int64,
    )

    group_pred = np.asarray(
        [
            LABEL_TO_ID[x]
            for x in group[
                "predicted_category"
            ]
        ],
        dtype=np.int64,
    )

    challenge_type_rows.append({
        "challenge_type":
            challenge_type,

        "count":
            len(group),

        "accuracy":
            float(
                accuracy_score(
                    group_true,
                    group_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    group_true,
                    group_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "review_rate":
            float(
                group[
                    "review_required"
                ].mean()
            ),

        "error_count":
            int(
                (
                    ~group[
                        "correct"
                    ].astype(bool)
                ).sum()
            ),
    })


challenge_type_df = pd.DataFrame(
    challenge_type_rows
).sort_values(
    "challenge_type"
)


# ============================================================
# OOD REVIEW / CLOSED-SET BEHAVIOR
# ============================================================

ood_df = (
    predictions_df[
        predictions_df[
            "challenge_group"
        ]
        == "OOD"
    ]
    .copy()
)


ood_review_count = int(
    ood_df[
        "review_required"
    ].sum()
)


ood_review_rate = (
    ood_review_count
    / len(ood_df)
)


ood_auto_accepted_count = (
    len(ood_df)
    - ood_review_count
)


ood_auto_accept_rate = (
    ood_auto_accepted_count
    / len(ood_df)
)


ood_family_rows = []


for (
    family,
    group,
) in ood_df.groupby(
    "ood_family"
):

    ood_family_rows.append({
        "ood_family":
            family,

        "count":
            len(group),

        "review_count":
            int(
                group[
                    "review_required"
                ].sum()
            ),

        "review_rate":
            float(
                group[
                    "review_required"
                ].mean()
            ),

        "auto_accepted_count":
            int(
                (
                    ~group[
                        "review_required"
                    ]
                ).sum()
            ),

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


ood_family_df = pd.DataFrame(
    ood_family_rows
).sort_values(
    "ood_family"
)


ood_prediction_distribution = (
    ood_df[
        "predicted_category"
    ]
    .value_counts()
    .rename_axis(
        "predicted_category"
    )
    .reset_index(
        name="count"
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


predictions_file = (
    REPORT_DIR
    / "14d_challenge_predictions.csv"
)

in_scope_metrics_file = (
    REPORT_DIR
    / "14d_in_scope_metrics.json"
)

confusion_file = (
    REPORT_DIR
    / "14d_in_scope_confusion_matrix.csv"
)

challenge_type_file = (
    REPORT_DIR
    / "14d_in_scope_by_challenge_type.csv"
)

ood_family_file = (
    REPORT_DIR
    / "14d_ood_by_family.csv"
)

ood_distribution_file = (
    REPORT_DIR
    / "14d_ood_prediction_distribution.csv"
)

summary_file = (
    REPORT_DIR
    / "14d_challenge_evaluation_summary.json"
)


predictions_df.to_csv(
    predictions_file,
    index=False,
)


pd.DataFrame(
    confusion,
    index=LABELS,
    columns=LABELS,
).to_csv(
    confusion_file
)


challenge_type_df.to_csv(
    challenge_type_file,
    index=False,
)


ood_family_df.to_csv(
    ood_family_file,
    index=False,
)


ood_prediction_distribution.to_csv(
    ood_distribution_file,
    index=False,
)


with in_scope_metrics_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "metrics":
                in_scope_metrics,

            "classification_report":
                classification_report_dict,

            "review_policy": {
                "top1_threshold":
                    top1_threshold,

                "margin_threshold":
                    margin_threshold,

                "review_rate":
                    in_scope_review_rate,

                "accepted_accuracy":
                    accepted_accuracy,

                "error_capture_rate":
                    error_capture_rate,

                "flagged_errors":
                    flagged_errors,

                "total_errors":
                    in_scope_error_count,
            },
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


summary = {
    "status":
        "CHALLENGE_EVALUATION_COMPLETE",

    "challenge_sha256":
        challenge_hash,

    "finetuned_model_sha256":
        finetuned_model_hash,

    "model":
        "full_finetuned_berturk",

    "device":
        str(device),

    "inference_seconds_150_records":
        elapsed_seconds,

    "review_policy": {
        "policy_id":
            review_policy[
                "policy_id"
            ],

        "top1_threshold":
            top1_threshold,

        "margin_threshold":
            margin_threshold,

        "threshold_source":
            "validation_only",

        "policy_frozen_before_challenge":
            True,
    },

    "in_scope": {
        "count":
            len(in_scope_df),

        **in_scope_metrics,

        "incorrect":
            in_scope_error_count,

        "review_count":
            in_scope_review_count,

        "review_rate":
            in_scope_review_rate,

        "accepted_count":
            len(
                accepted_in_scope
            ),

        "accepted_accuracy":
            accepted_accuracy,

        "flagged_errors":
            flagged_errors,

        "error_capture_rate":
            error_capture_rate,
    },

    "ood": {
        "count":
            len(ood_df),

        "review_count":
            ood_review_count,

        "review_rate":
            ood_review_rate,

        "auto_accepted_count":
            ood_auto_accepted_count,

        "auto_accept_rate":
            ood_auto_accept_rate,

        "interpretation":
            (
                "OOD has no class-accuracy metric. "
                "The closed-set classifier is "
                "evaluated only for whether the "
                "pre-frozen human-review policy "
                "flags the input for review."
            ),
    },

    "challenge_used_for_model_tuning":
        False,

    "challenge_used_for_threshold_tuning":
        False,

    "post_challenge_tuning_allowed":
        False,
}


with summary_file.open(
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
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print(
    "IN-SCOPE CHALLENGE RESULT"
)
print("=" * 80)

print(
    f"Accuracy            : "
    f"{in_scope_metrics['accuracy']:.6f}"
)

print(
    f"Macro Precision     : "
    f"{in_scope_metrics['macro_precision']:.6f}"
)

print(
    f"Macro Recall        : "
    f"{in_scope_metrics['macro_recall']:.6f}"
)

print(
    f"Macro-F1            : "
    f"{in_scope_metrics['macro_f1']:.6f}"
)

print(
    f"Weighted-F1         : "
    f"{in_scope_metrics['weighted_f1']:.6f}"
)

print(
    f"Log loss            : "
    f"{in_scope_metrics['log_loss']:.6f}"
)

print(
    f"Incorrect           : "
    f"{in_scope_error_count}/"
    f"{len(in_scope_df)}"
)

print()
print(
    f"Review count        : "
    f"{in_scope_review_count}"
)

print(
    f"Review rate         : "
    f"{in_scope_review_rate:.3%}"
)

print(
    f"Accepted count      : "
    f"{len(accepted_in_scope)}"
)

if accepted_accuracy is not None:

    print(
        f"Accepted accuracy   : "
        f"{accepted_accuracy:.3%}"
    )

else:

    print(
        "Accepted accuracy   : N/A"
    )


if (
    error_capture_rate
    is not None
):

    print(
        f"Errors caught review: "
        f"{flagged_errors}/"
        f"{in_scope_error_count}"
    )

    print(
        f"Error capture rate  : "
        f"{error_capture_rate:.3%}"
    )

else:

    print(
        "Error capture rate  : N/A "
        "(no challenge errors)"
    )


print()
print(
    "BY CHALLENGE TYPE"
)

for _, row in (
    challenge_type_df.iterrows()
):

    print(
        f"{row['challenge_type']:<24} "
        f"acc="
        f"{row['accuracy']:.3f} | "
        f"macroF1="
        f"{row['macro_f1']:.3f} | "
        f"review="
        f"{row['review_rate']:.3f} | "
        f"errors="
        f"{int(row['error_count'])}"
    )


print()
print("=" * 80)
print(
    "OOD / REVIEW-POLICY RESULT"
)
print("=" * 80)

print(
    f"OOD count           : "
    f"{len(ood_df)}"
)

print(
    f"Sent to review      : "
    f"{ood_review_count}"
)

print(
    f"OOD review rate     : "
    f"{ood_review_rate:.3%}"
)

print(
    f"Auto-accepted OOD   : "
    f"{ood_auto_accepted_count}"
)

print(
    f"OOD auto-accept rate: "
    f"{ood_auto_accept_rate:.3%}"
)


print()
print(
    "OOD BY FAMILY"
)

for _, row in (
    ood_family_df.iterrows()
):

    print(
        f"{row['ood_family']:<22} "
        f"review="
        f"{row['review_rate']:.3f} | "
        f"accepted="
        f"{int(row['auto_accepted_count'])}/"
        f"{int(row['count'])} | "
        f"mean_top1="
        f"{row['mean_top1_score']:.3f}"
    )


print()
print(
    "IMPORTANT:"
)

print(
    "OOD accuracy is intentionally "
    "not calculated."
)

print(
    "Closed-set model always predicts "
    "one of the five known classes."
)

print(
    "Challenge results must not be "
    "used for model or threshold tuning."
)


print()
print(
    f"Predictions          : "
    f"{predictions_file}"
)

print(
    f"Summary              : "
    f"{summary_file}"
)

print()
print(
    "STATUS: "
    "CHALLENGE_EVALUATION_COMPLETE"
)

