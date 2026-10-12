import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.pipeline import (
    FeatureUnion,
    Pipeline,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)


# ============================================================
# INPUTS
#
# IMPORTANT:
# TEST FILE IS INTENTIONALLY NOT DEFINED HERE.
#
# Stage 11A is TRAIN + VALIDATION only.
# ============================================================

TRAIN_FILE = Path(
    "data/splits/final/train.jsonl"
)

VAL_FILE = Path(
    "data/splits/final/validation.jsonl"
)


# ============================================================
# OUTPUTS
# ============================================================

REPORT_DIR = Path(
    "ml/reports/baseline"
)

MODEL_DIR = Path(
    "ml/models/baseline"
)

CANDIDATE_METRICS_FILE = REPORT_DIR / (
    "11a_baseline_candidate_metrics.csv"
)

BEST_CONFIG_FILE = REPORT_DIR / (
    "11a_best_baseline_config.json"
)

BEST_PER_CLASS_FILE = REPORT_DIR / (
    "11a_best_validation_per_class.csv"
)

BEST_CONFUSION_FILE = REPORT_DIR / (
    "11a_best_validation_confusion_matrix.csv"
)

SUMMARY_FILE = REPORT_DIR / (
    "11a_baseline_selection_summary.json"
)

BEST_MODEL_FILE = MODEL_DIR / (
    "tfidf_logreg_best_train_only.joblib"
)


# ============================================================
# FROZEN SPLIT HASHES
# ============================================================

EXPECTED_TRAIN_SHA256 = (
    "f53325aa45d45b3fbb8a0c356a1c838421e5b2860ab773f0301db5869e2426aa"
)

EXPECTED_VAL_SHA256 = (
    "7e22c40cb1fb62de9201e9ce5fddd14fa444cd63d3b27c6b765b03b687d99d75"
)


# ============================================================
# CONSTANTS
# ============================================================

RANDOM_SEED = 2026

EXPECTED_TRAIN_COUNT = 1167
EXPECTED_VAL_COUNT = 250

LABELS = [
    "TEMIZLIK_ATIK",
    "YOL_KALDIRIM",
    "PARK_YESIL_ALAN",
    "KENT_DUZENI_DENETIM",
    "HAYVAN_VETERINER",
]

C_VALUES = [
    0.5,
    1.0,
    2.0,
]


# ============================================================
# HELPERS
# ============================================================

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


def load_jsonl(path):

    records = []

    with path.open(
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

    return records


def aggregate_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        precision_macro,
        recall_macro,
        f1_macro,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy":
            float(accuracy),

        "precision_macro":
            float(
                precision_macro
            ),

        "recall_macro":
            float(
                recall_macro
            ),

        "f1_macro":
            float(
                f1_macro
            ),

        "precision_weighted":
            float(
                precision_weighted
            ),

        "recall_weighted":
            float(
                recall_weighted
            ),

        "f1_weighted":
            float(
                f1_weighted
            ),
    }


def create_features(
    feature_name,
):

    if feature_name == "WORD":

        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=40000,
            sublinear_tf=True,
            strip_accents=None,
            lowercase=True,
            norm="l2",
        )


    if feature_name == "CHAR":

        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
            lowercase=True,
            norm="l2",
        )


    if feature_name == "HYBRID":

        return FeatureUnion([
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=30000,
                    sublinear_tf=True,
                    lowercase=True,
                    norm="l2",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=40000,
                    sublinear_tf=True,
                    lowercase=True,
                    norm="l2",
                ),
            ),
        ])


    raise ValueError(
        f"Unknown feature set: "
        f"{feature_name}"
    )


def build_pipeline(
    feature_name,
    c_value,
):

    return Pipeline([
        (
            "features",
            create_features(
                feature_name
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=3000,
                random_state=RANDOM_SEED,
                class_weight=None,
            ),
        ),
    ])


# ============================================================
# VERIFY HASHES
# ============================================================

train_hash = sha256_file(
    TRAIN_FILE
)

val_hash = sha256_file(
    VAL_FILE
)


if (
    train_hash
    != EXPECTED_TRAIN_SHA256
):

    raise ValueError(
        "FINAL TRAIN hash changed!\n"
        f"Expected: "
        f"{EXPECTED_TRAIN_SHA256}\n"
        f"Actual  : "
        f"{train_hash}"
    )


if (
    val_hash
    != EXPECTED_VAL_SHA256
):

    raise ValueError(
        "FINAL VALIDATION hash changed!\n"
        f"Expected: "
        f"{EXPECTED_VAL_SHA256}\n"
        f"Actual  : "
        f"{val_hash}"
    )


# ============================================================
# LOAD DATA
# ============================================================

train_records = load_jsonl(
    TRAIN_FILE
)

val_records = load_jsonl(
    VAL_FILE
)


if (
    len(train_records)
    != EXPECTED_TRAIN_COUNT
):

    raise ValueError(
        f"Expected train "
        f"{EXPECTED_TRAIN_COUNT}, "
        f"found "
        f"{len(train_records)}."
    )


if (
    len(val_records)
    != EXPECTED_VAL_COUNT
):

    raise ValueError(
        f"Expected validation "
        f"{EXPECTED_VAL_COUNT}, "
        f"found "
        f"{len(val_records)}."
    )


train_ids = {
    r["sample_id"]
    for r in train_records
}

val_ids = {
    r["sample_id"]
    for r in val_records
}


if train_ids & val_ids:

    raise ValueError(
        "TRAIN / VALIDATION "
        "sample overlap detected."
    )


# ============================================================
# EXTRACT MODEL INPUT / TARGET
# ============================================================

X_train = [
    record["complaint_text"]
    for record in train_records
]

y_train = [
    record["category"]
    for record in train_records
]

X_val = [
    record["complaint_text"]
    for record in val_records
]

y_val = [
    record["category"]
    for record in val_records
]


unknown_train_labels = (
    set(y_train)
    - set(LABELS)
)

unknown_val_labels = (
    set(y_val)
    - set(LABELS)
)


if (
    unknown_train_labels
    or unknown_val_labels
):

    raise ValueError(
        "Unknown category label found."
    )


if set(y_train) != set(LABELS):

    raise ValueError(
        "Not all labels occur in TRAIN."
    )


if set(y_val) != set(LABELS):

    raise ValueError(
        "Not all labels occur in VALIDATION."
    )


# ============================================================
# CREATE OUTPUT DIRS
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL SELECTION
# ============================================================

feature_sets = [
    "WORD",
    "CHAR",
    "HYBRID",
]


candidate_results = []

best_pipeline = None
best_result = None


print("=" * 78)
print(
    "STAGE 11A - TF-IDF + LOGISTIC "
    "REGRESSION BASELINE SELECTION"
)
print("=" * 78)

print(
    f"TRAIN records      : "
    f"{len(train_records)}"
)

print(
    f"VALIDATION records : "
    f"{len(val_records)}"
)

print(
    "Selection metric   : "
    "VALIDATION MACRO-F1"
)

print(
    "TEST accessed      : False"
)


for feature_name in feature_sets:

    for c_value in C_VALUES:

        candidate_id = (
            f"{feature_name}_C"
            + str(c_value).replace(
                ".",
                "p",
            )
        )


        print(
            f"\nTraining "
            f"{candidate_id} ..."
        )


        pipeline = build_pipeline(
            feature_name,
            c_value,
        )


        start = time.perf_counter()


        pipeline.fit(
            X_train,
            y_train,
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        train_pred = pipeline.predict(
            X_train
        )

        val_pred = pipeline.predict(
            X_val
        )


        train_proba = (
            pipeline.predict_proba(
                X_train
            )
        )

        val_proba = (
            pipeline.predict_proba(
                X_val
            )
        )


        train_metrics = (
            aggregate_metrics(
                y_train,
                train_pred,
            )
        )

        val_metrics = (
            aggregate_metrics(
                y_val,
                val_pred,
            )
        )


        classifier_classes = (
            pipeline.named_steps[
                "classifier"
            ].classes_
        )


        train_ll = log_loss(
            y_train,
            train_proba,
            labels=classifier_classes,
        )

        val_ll = log_loss(
            y_val,
            val_proba,
            labels=classifier_classes,
        )


        result = {
            "candidate_id":
                candidate_id,

            "feature_set":
                feature_name,

            "C":
                c_value,

            "training_seconds":
                float(elapsed),

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "train_macro_f1":
                train_metrics[
                    "f1_macro"
                ],

            "train_weighted_f1":
                train_metrics[
                    "f1_weighted"
                ],

            "train_log_loss":
                float(
                    train_ll
                ),

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_precision_macro":
                val_metrics[
                    "precision_macro"
                ],

            "val_recall_macro":
                val_metrics[
                    "recall_macro"
                ],

            "val_macro_f1":
                val_metrics[
                    "f1_macro"
                ],

            "val_weighted_f1":
                val_metrics[
                    "f1_weighted"
                ],

            "val_log_loss":
                float(
                    val_ll
                ),
        }


        candidate_results.append(
            result
        )


        print(
            f"  train Macro-F1 : "
            f"{result['train_macro_f1']:.4f}"
        )

        print(
            f"  val Macro-F1   : "
            f"{result['val_macro_f1']:.4f}"
        )

        print(
            f"  val Accuracy   : "
            f"{result['val_accuracy']:.4f}"
        )

        print(
            f"  val Log Loss   : "
            f"{result['val_log_loss']:.4f}"
        )


        # ----------------------------------------------------
        # Selection policy:
        #
        # 1. higher validation Macro-F1
        # 2. lower validation log-loss
        # 3. higher validation accuracy
        # ----------------------------------------------------

        if best_result is None:

            is_better = True

        else:

            candidate_key = (
                result[
                    "val_macro_f1"
                ],
                -result[
                    "val_log_loss"
                ],
                result[
                    "val_accuracy"
                ],
            )

            best_key = (
                best_result[
                    "val_macro_f1"
                ],
                -best_result[
                    "val_log_loss"
                ],
                best_result[
                    "val_accuracy"
                ],
            )

            is_better = (
                candidate_key
                > best_key
            )


        if is_better:

            best_result = dict(
                result
            )

            best_pipeline = pipeline


# ============================================================
# VALIDATION DETAIL FOR BEST MODEL
# ============================================================

if (
    best_pipeline is None
    or best_result is None
):

    raise RuntimeError(
        "No best baseline selected."
    )


best_val_pred = (
    best_pipeline.predict(
        X_val
    )
)


(
    per_precision,
    per_recall,
    per_f1,
    per_support,
) = precision_recall_fscore_support(
    y_val,
    best_val_pred,
    labels=LABELS,
    average=None,
    zero_division=0,
)


per_class_rows = []


for (
    label,
    precision,
    recall,
    f1,
    support,
) in zip(
    LABELS,
    per_precision,
    per_recall,
    per_f1,
    per_support,
):

    per_class_rows.append({
        "category":
            label,

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "support":
            int(
                support
            ),
    })


cm = confusion_matrix(
    y_val,
    best_val_pred,
    labels=LABELS,
)


# ============================================================
# SAVE BEST PIPELINE
# ============================================================

joblib.dump(
    best_pipeline,
    BEST_MODEL_FILE,
)


best_model_hash = sha256_file(
    BEST_MODEL_FILE
)


# ============================================================
# WRITE CANDIDATE METRICS
# ============================================================

candidate_fields = [
    "candidate_id",
    "feature_set",
    "C",
    "training_seconds",

    "train_accuracy",
    "train_macro_f1",
    "train_weighted_f1",
    "train_log_loss",

    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_macro_f1",
    "val_weighted_f1",
    "val_log_loss",
]


with CANDIDATE_METRICS_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=candidate_fields,
    )

    writer.writeheader()

    writer.writerows(
        candidate_results
    )


# ============================================================
# WRITE PER-CLASS VALIDATION METRICS
# ============================================================

with BEST_PER_CLASS_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "precision",
            "recall",
            "f1",
            "support",
        ],
    )

    writer.writeheader()

    writer.writerows(
        per_class_rows
    )


# ============================================================
# WRITE CONFUSION MATRIX
# ============================================================

with BEST_CONFUSION_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["actual\\predicted"]
        + LABELS
    )

    for label, row in zip(
        LABELS,
        cm,
    ):

        writer.writerow(
            [label]
            + [
                int(value)
                for value in row
            ]
        )


# ============================================================
# BEST CONFIG
# ============================================================

best_config = {
    "candidate_id":
        best_result[
            "candidate_id"
        ],

    "feature_set":
        best_result[
            "feature_set"
        ],

    "logistic_regression_C":
        best_result[
            "C"
        ],

    "selection_metric":
        "validation_macro_f1",

    "selection_tie_breakers": [
        "lower_validation_log_loss",
        "higher_validation_accuracy",
    ],

    "validation_metrics": {
        "accuracy":
            best_result[
                "val_accuracy"
            ],

        "precision_macro":
            best_result[
                "val_precision_macro"
            ],

        "recall_macro":
            best_result[
                "val_recall_macro"
            ],

        "macro_f1":
            best_result[
                "val_macro_f1"
            ],

        "weighted_f1":
            best_result[
                "val_weighted_f1"
            ],

        "log_loss":
            best_result[
                "val_log_loss"
            ],
    },

    "train_metrics": {
        "accuracy":
            best_result[
                "train_accuracy"
            ],

        "macro_f1":
            best_result[
                "train_macro_f1"
            ],

        "weighted_f1":
            best_result[
                "train_weighted_f1"
            ],

        "log_loss":
            best_result[
                "train_log_loss"
            ],
    },

    "model_file":
        str(
            BEST_MODEL_FILE
        ),

    "model_sha256":
        best_model_hash,

    "test_accessed":
        False,
}


with BEST_CONFIG_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        best_config,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "status":
        "BASELINE_VALIDATION_SELECTION_COMPLETE",

    "experiment":
        (
            "TF-IDF + Logistic Regression"
        ),

    "train_count":
        len(train_records),

    "validation_count":
        len(val_records),

    "train_sha256":
        train_hash,

    "validation_sha256":
        val_hash,

    "candidate_count":
        len(
            candidate_results
        ),

    "feature_sets_tested":
        feature_sets,

    "C_values_tested":
        C_VALUES,

    "class_weight":
        None,

    "random_seed":
        RANDOM_SEED,

    "primary_selection_metric":
        "validation_macro_f1",

    "best_candidate":
        best_result,

    "best_model_file":
        str(
            BEST_MODEL_FILE
        ),

    "best_model_sha256":
        best_model_hash,

    "test_accessed":
        False,

    "test_metrics_reported":
        False,

    "policy": (
        "Model and hyperparameter selection "
        "used only the frozen TRAIN and "
        "VALIDATION splits. The frozen TEST "
        "split was intentionally not loaded "
        "or accessed during Stage 11A."
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
# TERMINAL
# ============================================================

print("\n" + "=" * 78)
print("BASELINE SELECTION RESULT")
print("=" * 78)

print(
    f"Candidates evaluated : "
    f"{len(candidate_results)}"
)

print(
    f"Best candidate       : "
    f"{best_result['candidate_id']}"
)

print(
    f"Feature set          : "
    f"{best_result['feature_set']}"
)

print(
    f"C                    : "
    f"{best_result['C']}"
)


print("\nBest TRAIN metrics:")

print(
    f"  Accuracy           : "
    f"{best_result['train_accuracy']:.4f}"
)

print(
    f"  Macro-F1           : "
    f"{best_result['train_macro_f1']:.4f}"
)

print(
    f"  Log Loss           : "
    f"{best_result['train_log_loss']:.4f}"
)


print("\nBest VALIDATION metrics:")

print(
    f"  Accuracy           : "
    f"{best_result['val_accuracy']:.4f}"
)

print(
    f"  Precision Macro    : "
    f"{best_result['val_precision_macro']:.4f}"
)

print(
    f"  Recall Macro       : "
    f"{best_result['val_recall_macro']:.4f}"
)

print(
    f"  Macro-F1           : "
    f"{best_result['val_macro_f1']:.4f}"
)

print(
    f"  Weighted-F1        : "
    f"{best_result['val_weighted_f1']:.4f}"
)

print(
    f"  Log Loss           : "
    f"{best_result['val_log_loss']:.4f}"
)


print("\nModel:")

print(
    f"  Path               : "
    f"{BEST_MODEL_FILE}"
)

print(
    f"  SHA256             : "
    f"{best_model_hash}"
)


print("\nIMPORTANT:")

print(
    "  Frozen TEST was NOT accessed."
)

print(
    "  Stage 11B will perform the "
    "single final baseline test."
)


print(
    "\nSTATUS: "
    "BASELINE_VALIDATION_SELECTION_COMPLETE"
)