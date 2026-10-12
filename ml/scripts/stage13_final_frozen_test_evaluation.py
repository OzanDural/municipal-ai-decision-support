import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
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
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# CONFIG
# ============================================================

TEST_FILE = Path(
    "data/splits/final/test.jsonl"
)

BASELINE_MODEL_FILE = Path(
    "ml/models/baseline/"
    "tfidf_logreg_best_train_only.joblib"
)

FROZEN_PROBE_FILE = Path(
    "ml/models/frozen_berturk/"
    "frozen_berturk_linear_probe.joblib"
)

FINETUNED_MODEL_DIR = Path(
    "ml/models/berturk/final_model"
)

FINETUNED_MODEL_WEIGHTS = (
    FINETUNED_MODEL_DIR
    / "model.safetensors"
)

REPORT_DIR = Path(
    "ml/reports/final_test"
)


EXPECTED_TEST_SHA256 = (
    "59d69c1995c377158ccee355962a2ec"
    "152e2cda754fc478bfcf71922969aea69"
)

EXPECTED_BASELINE_SHA256 = (
    "5c2088ef51dbe3a4ea799c36731a22c"
    "6a6bfcd2eccd9c1cf9e6cdb67ac60cd9d"
)

EXPECTED_FINETUNED_MODEL_SHA256 = (
    "af3aaf2d03972e0b07f5858971b9114b"
    "efea21b7127e390d088e3c85e7bd1bb4"
)


BASE_MODEL_NAME = (
    "dbmdz/bert-base-turkish-cased"
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

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
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
            "otomatik belirlenemedi."
        )

    return candidates[0].parent


def extract_estimator(obj):

    if (
        hasattr(obj, "predict")
        and hasattr(obj, "predict_proba")
    ):

        return obj

    if isinstance(obj, dict):

        candidate_keys = [
            "pipeline",
            "model",
            "estimator",
            "best_model",
            "classifier",
        ]

        for key in candidate_keys:

            value = obj.get(key)

            if (
                value is not None
                and hasattr(
                    value,
                    "predict",
                )
                and hasattr(
                    value,
                    "predict_proba",
                )
            ):

                return value

    raise RuntimeError(
        "Joblib içinden kullanılabilir "
        "predict/predict_proba modeli "
        "bulunamadı."
    )


def class_value_to_id(value):
    """Convert sklearn class values to the project's canonical integer IDs."""
    if isinstance(value, (str, np.str_)):
        value = str(value)
        if value not in LABEL_TO_ID:
            raise RuntimeError(f"Unknown string class: {value}")
        return LABEL_TO_ID[value]

    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value not in ID_TO_LABEL:
            raise RuntimeError(f"Unknown numeric class: {value}")
        return value

    raise RuntimeError(
        f"Unsupported class type: {type(value)} -> {value}"
    )


def get_estimator_classes(estimator):
    """Retrieve sklearn classes_ from either a direct estimator or Pipeline."""
    classes = getattr(estimator, "classes_", None)

    if classes is None and hasattr(estimator, "named_steps"):
        final_estimator = list(estimator.named_steps.values())[-1]
        classes = getattr(final_estimator, "classes_", None)

    if classes is None:
        raise RuntimeError(
            "Estimator classes_ could not be determined."
        )

    return classes


def canonicalize_sklearn_outputs(
    estimator,
    raw_predictions,
    raw_probabilities,
):
    """
    Convert sklearn predictions and predict_proba columns to the fixed
    project label order:
      0 TEMIZLIK_ATIK
      1 YOL_KALDIRIM
      2 PARK_YESIL_ALAN
      3 KENT_DUZENI_DENETIM
      4 HAYVAN_VETERINER
    """
    classes = get_estimator_classes(estimator)

    class_ids = [
        class_value_to_id(x)
        for x in classes
    ]

    expected_ids = set(range(len(LABELS)))

    if set(class_ids) != expected_ids:
        raise RuntimeError(
            "Estimator class set does not match canonical labels.\n"
            f"Estimator classes: {classes}\n"
            f"Mapped ids       : {class_ids}"
        )

    predictions = np.asarray(
        [class_value_to_id(x) for x in raw_predictions],
        dtype=np.int64,
    )

    raw_probabilities = np.asarray(
        raw_probabilities,
        dtype=float,
    )

    if raw_probabilities.ndim != 2:
        raise RuntimeError(
            "predict_proba() must return a 2-D array, "
            f"found shape {raw_probabilities.shape}"
        )

    if raw_probabilities.shape[1] != len(LABELS):
        raise RuntimeError(
            "Unexpected probability shape: "
            f"{raw_probabilities.shape}"
        )

    probabilities = np.zeros_like(
        raw_probabilities,
        dtype=float,
    )

    for source_column, class_id in enumerate(class_ids):
        probabilities[:, class_id] = raw_probabilities[:, source_column]

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise RuntimeError(
            "Canonicalized probabilities do not sum to 1."
        )

    probability_predictions = np.argmax(
        probabilities,
        axis=1,
    )

    if not np.array_equal(predictions, probability_predictions):
        raise RuntimeError(
            "predict() and predict_proba() class ordering are inconsistent."
        )

    return predictions, probabilities


def metrics_dict(
    y_true,
    y_pred,
    probabilities,
):

    result = {
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

    return result


def save_evaluation(
    model_name,
    y_true,
    y_pred,
    probabilities,
    records,
    elapsed_seconds,
):

    model_slug = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_dir = (
        REPORT_DIR
        / model_slug
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    metrics = metrics_dict(
        y_true,
        y_pred,
        probabilities,
    )


    report = classification_report(
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


    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(
                len(LABELS)
            )
        ),
    )


    with (
        model_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "model":
                    model_name,

                "metrics":
                    metrics,

                "elapsed_seconds":
                    elapsed_seconds,

                "classification_report":
                    report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


    pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS,
    ).to_csv(
        model_dir
        / "confusion_matrix.csv"
    )


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

        prediction_rows.append({
            "sample_id":
                record["sample_id"],

            "complaint_text":
                record[
                    "complaint_text"
                ],

            "true_category":
                record["category"],

            "predicted_category":
                ID_TO_LABEL[
                    int(y_pred[i])
                ],

            "correct":
                bool(
                    int(y_pred[i])
                    == int(y_true[i])
                ),

            "top1_category":
                ID_TO_LABEL[
                    top1_id
                ],

            "top1_score":
                float(
                    probs[
                        top1_id
                    ]
                ),

            "top2_category":
                ID_TO_LABEL[
                    top2_id
                ],

            "top2_score":
                float(
                    probs[
                        top2_id
                    ]
                ),

            "score_margin":
                float(
                    probs[
                        top1_id
                    ]
                    -
                    probs[
                        top2_id
                    ]
                ),

            **{
                f"prob_{LABELS[j]}":
                    float(
                        probs[j]
                    )
                for j in range(
                    len(LABELS)
                )
            },
        })


    pd.DataFrame(
        prediction_rows
    ).to_csv(
        model_dir
        / "predictions.csv",
        index=False,
    )


    return {
        "model":
            model_name,

        **metrics,

        "elapsed_seconds":
            elapsed_seconds,

        "correct":
            int(
                np.sum(
                    np.asarray(
                        y_true
                    )
                    ==
                    np.asarray(
                        y_pred
                    )
                )
            ),

        "incorrect":
            int(
                np.sum(
                    np.asarray(
                        y_true
                    )
                    !=
                    np.asarray(
                        y_pred
                    )
                )
            ),
    }


# ============================================================
# VERIFY INPUTS
# ============================================================

required_paths = [
    TEST_FILE,
    BASELINE_MODEL_FILE,
    FROZEN_PROBE_FILE,
    FINETUNED_MODEL_DIR,
    FINETUNED_MODEL_WEIGHTS,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required path:\n"
            f"{path}"
        )


test_hash = sha256_file(
    TEST_FILE
)

baseline_hash = sha256_file(
    BASELINE_MODEL_FILE
)

finetuned_model_hash = (
    sha256_file(
        FINETUNED_MODEL_WEIGHTS
    )
)


if test_hash != EXPECTED_TEST_SHA256:

    raise RuntimeError(
        "FINAL TEST SHA256 mismatch.\n"
        f"Expected: "
        f"{EXPECTED_TEST_SHA256}\n"
        f"Found   : "
        f"{test_hash}"
    )


if (
    baseline_hash
    != EXPECTED_BASELINE_SHA256
):

    raise RuntimeError(
        "Baseline model SHA256 "
        "mismatch.\n"
        f"Expected: "
        f"{EXPECTED_BASELINE_SHA256}\n"
        f"Found   : "
        f"{baseline_hash}"
    )


if (
    finetuned_model_hash
    != EXPECTED_FINETUNED_MODEL_SHA256
):

    raise RuntimeError(
        "Fine-tuned BERTurk model "
        "SHA256 mismatch.\n"
        f"Expected: "
        f"{EXPECTED_FINETUNED_MODEL_SHA256}\n"
        f"Found   : "
        f"{finetuned_model_hash}"
    )


# ============================================================
# LOAD TEST
# ============================================================

records = load_jsonl(
    TEST_FILE
)


if len(records) != 250:

    raise RuntimeError(
        f"Expected 250 final TEST "
        f"records, found {len(records)}."
    )


for record in records:

    if (
        record.get("category")
        not in LABEL_TO_ID
    ):

        raise RuntimeError(
            "Invalid category in TEST."
        )


texts = [
    record[
        "complaint_text"
    ]
    for record in records
]


y_true = np.asarray(
    [
        LABEL_TO_ID[
            record["category"]
        ]
        for record in records
    ],
    dtype=np.int64,
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 80)
print(
    "STAGE 13 - FINAL FROZEN "
    "TEST EVALUATION"
)
print("=" * 80)

print(
    f"TEST records       : "
    f"{len(records)}"
)

print(
    f"TEST SHA-256       : "
    f"{test_hash}"
)

print(
    "TEST hash          : PASS"
)

print(
    "Baseline hash      : PASS"
)

print(
    "Fine-tuned model hash: PASS"
)

print()
print(
    "IMPORTANT: FINAL TEST IS "
    "NOW BEING ACCESSED."
)

print(
    "No post-test model tuning "
    "is permitted."
)


# ============================================================
# 1. TF-IDF + LOGISTIC REGRESSION
# ============================================================

print()
print("=" * 80)
print(
    "MODEL 1/3 - TF-IDF + "
    "LOGISTIC REGRESSION"
)
print("=" * 80)


baseline_obj = joblib.load(
    BASELINE_MODEL_FILE
)

baseline_model = (
    extract_estimator(
        baseline_obj
    )
)


start = time.perf_counter()

baseline_pred_raw = (
    baseline_model.predict(
        texts
    )
)

baseline_prob_raw = (
    baseline_model.predict_proba(
        texts
    )
)

baseline_seconds = (
    time.perf_counter()
    - start
)


baseline_pred, baseline_prob = (
    canonicalize_sklearn_outputs(
        baseline_model,
        baseline_pred_raw,
        baseline_prob_raw,
    )
)


baseline_result = save_evaluation(
    "TFIDF_LOGREG",
    y_true,
    baseline_pred,
    baseline_prob,
    records,
    baseline_seconds,
)


print(
    f"Accuracy            : "
    f"{baseline_result['accuracy']:.6f}"
)

print(
    f"Macro-F1            : "
    f"{baseline_result['macro_f1']:.6f}"
)

print(
    f"Incorrect           : "
    f"{baseline_result['incorrect']}"
)


# ============================================================
# 2. FROZEN BERTURK + LINEAR PROBE
# ============================================================

print()
print("=" * 80)
print(
    "MODEL 2/3 - FROZEN BERTURK "
    "+ LINEAR PROBE"
)
print("=" * 80)


frozen_obj = joblib.load(
    FROZEN_PROBE_FILE
)

frozen_probe = (
    extract_estimator(
        frozen_obj
    )
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        BASE_MODEL_NAME
    )
)

base_encoder = (
    AutoModel
    .from_pretrained(
        BASE_MODEL_NAME
    )
)

base_encoder.to(
    device
)

base_encoder.eval()


for parameter in (
    base_encoder.parameters()
):

    parameter.requires_grad = False


if any(
    p.requires_grad
    for p in base_encoder.parameters()
):

    raise RuntimeError(
        "Base BERTurk encoder "
        "is not fully frozen."
    )


embedding_parts = []


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

        outputs = base_encoder(
            **encoded
        )

        embeddings = (
            outputs.pooler_output
            .detach()
            .cpu()
            .numpy()
        )

        embedding_parts.append(
            embeddings
        )


X_test_frozen = np.concatenate(
    embedding_parts,
    axis=0,
)


frozen_pred_raw = (
    frozen_probe.predict(
        X_test_frozen
    )
)

frozen_prob_raw = (
    frozen_probe.predict_proba(
        X_test_frozen
    )
)


frozen_seconds = (
    time.perf_counter()
    - start
)


frozen_pred, frozen_prob = (
    canonicalize_sklearn_outputs(
        frozen_probe,
        frozen_pred_raw,
        frozen_prob_raw,
    )
)


frozen_result = save_evaluation(
    "FROZEN_BERTURK_LINEAR_PROBE",
    y_true,
    frozen_pred,
    frozen_prob,
    records,
    frozen_seconds,
)


print(
    f"Accuracy            : "
    f"{frozen_result['accuracy']:.6f}"
)

print(
    f"Macro-F1            : "
    f"{frozen_result['macro_f1']:.6f}"
)

print(
    f"Incorrect           : "
    f"{frozen_result['incorrect']}"
)


del base_encoder

if device.type == "cuda":

    torch.cuda.empty_cache()


# ============================================================
# 3. FULL FINE-TUNED BERTURK
# ============================================================

print()
print("=" * 80)
print(
    "MODEL 3/3 - FULL "
    "FINE-TUNED BERTURK"
)
print("=" * 80)


finetuned_root = find_model_root(
    FINETUNED_MODEL_DIR
)


ft_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        finetuned_root
    )
)

ft_model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        finetuned_root
    )
)

ft_model.to(
    device
)

ft_model.eval()


print(
    f"Model root          : "
    f"{finetuned_root}"
)

print(
    f"Device              : "
    f"{device}"
)


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

        encoded = ft_tokenizer(
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

        outputs = ft_model(
            **encoded
        )

        all_logits.append(
            outputs.logits
            .detach()
            .cpu()
        )


logits = torch.cat(
    all_logits,
    dim=0,
)


finetuned_prob = (
    torch.softmax(
        logits,
        dim=-1,
    )
    .numpy()
)


finetuned_pred = (
    np.argmax(
        finetuned_prob,
        axis=1,
    )
    .astype(
        np.int64
    )
)


finetuned_seconds = (
    time.perf_counter()
    - start
)


finetuned_result = save_evaluation(
    "FULL_FINETUNED_BERTURK",
    y_true,
    finetuned_pred,
    finetuned_prob,
    records,
    finetuned_seconds,
)


print(
    f"Accuracy            : "
    f"{finetuned_result['accuracy']:.6f}"
)

print(
    f"Macro-F1            : "
    f"{finetuned_result['macro_f1']:.6f}"
)

print(
    f"Incorrect           : "
    f"{finetuned_result['incorrect']}"
)


# ============================================================
# COMPARISON TABLE
# ============================================================

results = [
    baseline_result,
    frozen_result,
    finetuned_result,
]


comparison_df = pd.DataFrame(
    results
)


comparison_df.to_csv(
    REPORT_DIR
    / "13_final_model_comparison.csv",
    index=False,
)


# ============================================================
# AGREEMENT / DISAGREEMENT ANALYSIS
# ============================================================

agreement_df = pd.DataFrame({
    "sample_id":
        [
            r["sample_id"]
            for r in records
        ],

    "true_category":
        [
            r["category"]
            for r in records
        ],

    "tfidf_prediction":
        [
            ID_TO_LABEL[int(x)]
            for x in baseline_pred
        ],

    "frozen_berturk_prediction":
        [
            ID_TO_LABEL[int(x)]
            for x in frozen_pred
        ],

    "finetuned_berturk_prediction":
        [
            ID_TO_LABEL[int(x)]
            for x in finetuned_pred
        ],
})


agreement_df[
    "all_three_agree"
] = (
    (
        agreement_df[
            "tfidf_prediction"
        ]
        ==
        agreement_df[
            "frozen_berturk_prediction"
        ]
    )
    &
    (
        agreement_df[
            "tfidf_prediction"
        ]
        ==
        agreement_df[
            "finetuned_berturk_prediction"
        ]
    )
)


agreement_df[
    "finetuned_correct"
] = (
    agreement_df[
        "finetuned_berturk_prediction"
    ]
    ==
    agreement_df[
        "true_category"
    ]
)


agreement_df.to_csv(
    REPORT_DIR
    / "13_model_agreement.csv",
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "status":
        "FINAL_FROZEN_TEST_COMPLETE",

    "test_frozen_before_run":
        True,

    "test_accessed":
        True,

    "test_size":
        len(records),

    "test_sha256":
        test_hash,

    "baseline_model_sha256":
        baseline_hash,

    "finetuned_model_sha256":
        finetuned_model_hash,

    "models": {
        "tfidf_logreg":
            baseline_result,

        "frozen_berturk_linear_probe":
            frozen_result,

        "full_finetuned_berturk":
            finetuned_result,
    },

    "post_test_tuning_allowed":
        False,

    "note":
        (
            "All three locked model "
            "configurations were evaluated "
            "on exactly the same frozen final "
            "test set. Results must not be "
            "used for post-hoc tuning."
        ),
}


with (
    REPORT_DIR
    / "13_final_frozen_test_summary.json"
).open(
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
    "FINAL FROZEN TEST COMPARISON"
)
print("=" * 80)

for result in results:

    print()
    print(
        result["model"]
    )

    print(
        f"  Accuracy        : "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"  Macro Precision : "
        f"{result['macro_precision']:.6f}"
    )

    print(
        f"  Macro Recall    : "
        f"{result['macro_recall']:.6f}"
    )

    print(
        f"  Macro-F1        : "
        f"{result['macro_f1']:.6f}"
    )

    print(
        f"  Weighted-F1     : "
        f"{result['weighted_f1']:.6f}"
    )

    print(
        f"  Log loss        : "
        f"{result['log_loss']:.6f}"
    )

    print(
        f"  Correct         : "
        f"{result['correct']}/"
        f"{len(records)}"
    )

    print(
        f"  Incorrect       : "
        f"{result['incorrect']}"
    )


print()
print(
    "Final TEST SHA256 : "
    f"{test_hash}"
)

print(
    "Post-test tuning   : "
    "NOT ALLOWED"
)

print()
print(
    "Outputs            : "
    f"{REPORT_DIR}"
)

print()
print(
    "STATUS: "
    "FINAL_FROZEN_TEST_COMPLETE"
)

