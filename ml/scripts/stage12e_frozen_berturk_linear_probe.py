import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from transformers import AutoModel, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "dbmdz/bert-base-turkish-cased"

TRAIN_FILE = Path(
    "data/splits/final/train.jsonl"
)

VAL_FILE = Path(
    "data/splits/final/validation.jsonl"
)

MODEL_DIR = Path(
    "ml/models/frozen_berturk"
)

REPORT_DIR = Path(
    "ml/reports/model_evaluation"
)

MODEL_OUTPUT = (
    MODEL_DIR
    / "frozen_berturk_linear_probe.joblib"
)

SUMMARY_OUTPUT = (
    REPORT_DIR
    / "12e_frozen_berturk_linear_probe_summary.json"
)

CANDIDATES_OUTPUT = (
    REPORT_DIR
    / "12e_frozen_berturk_candidates.csv"
)

TRAIN_EMBEDDINGS_OUTPUT = (
    MODEL_DIR
    / "train_embeddings.npy"
)

VAL_EMBEDDINGS_OUTPUT = (
    MODEL_DIR
    / "validation_embeddings.npy"
)


EXPECTED_TRAIN_SHA256 = (
    "f53325aa45d45b3fbb8a0c356a1c838421e5b2860ab773f0301db5869e2426aa"
)

EXPECTED_VAL_SHA256 = (
    "7e22c40cb1fb62de9201e9ce5fddd14fa444cd63d3b27c6b765b03b687d99d75"
)


MAX_LENGTH = 160

BATCH_SIZE = 8

RANDOM_SEED = 2026

C_VALUES = [
    0.5,
    1.0,
    2.0,
]


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


def validate_records(records, split_name):

    for i, record in enumerate(records):

        if "complaint_text" not in record:
            raise RuntimeError(
                f"{split_name}: complaint_text missing "
                f"at record {i}"
            )

        if "category" not in record:
            raise RuntimeError(
                f"{split_name}: category missing "
                f"at record {i}"
            )

        if record["category"] not in LABEL_TO_ID:
            raise RuntimeError(
                f"{split_name}: invalid label "
                f"{record['category']}"
            )


def extract_embeddings(
    texts,
    tokenizer,
    model,
    device,
    split_name,
):

    all_embeddings = []

    total_batches = (
        len(texts)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    start_time = time.perf_counter()

    with torch.inference_mode():

        for batch_index in range(
            0,
            len(texts),
            BATCH_SIZE,
        ):

            batch_texts = texts[
                batch_index:
                batch_index + BATCH_SIZE
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value
                in encoded.items()
            }

            outputs = model(
                **encoded
            )

            # Hugging Face BERT sequence-classification
            # head uses pooled CLS representation.
            #
            # We use the pretrained pooled representation
            # while keeping the entire encoder frozen.
            embeddings = (
                outputs.pooler_output
                .detach()
                .cpu()
                .numpy()
            )

            all_embeddings.append(
                embeddings
            )

            current_batch = (
                batch_index
                // BATCH_SIZE
                + 1
            )

            if (
                current_batch == 1
                or current_batch % 10 == 0
                or current_batch == total_batches
            ):

                if device.type == "cuda":

                    allocated_mb = (
                        torch.cuda.memory_allocated()
                        / 1024**2
                    )

                    print(
                        f"{split_name}: "
                        f"{current_batch}/"
                        f"{total_batches} batches | "
                        f"GPU allocated "
                        f"{allocated_mb:.1f} MB"
                    )

                else:

                    print(
                        f"{split_name}: "
                        f"{current_batch}/"
                        f"{total_batches} batches"
                    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    matrix = np.concatenate(
        all_embeddings,
        axis=0,
    )

    return matrix, elapsed


def calculate_metrics(
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
# INPUT VALIDATION
# ============================================================

for path in [
    TRAIN_FILE,
    VAL_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )


train_hash = sha256_file(
    TRAIN_FILE
)

val_hash = sha256_file(
    VAL_FILE
)


if train_hash != EXPECTED_TRAIN_SHA256:

    raise RuntimeError(
        "TRAIN hash mismatch.\n"
        f"Expected: {EXPECTED_TRAIN_SHA256}\n"
        f"Found   : {train_hash}"
    )


if val_hash != EXPECTED_VAL_SHA256:

    raise RuntimeError(
        "VALIDATION hash mismatch.\n"
        f"Expected: {EXPECTED_VAL_SHA256}\n"
        f"Found   : {val_hash}"
    )


train_records = load_jsonl(
    TRAIN_FILE
)

val_records = load_jsonl(
    VAL_FILE
)


validate_records(
    train_records,
    "TRAIN",
)

validate_records(
    val_records,
    "VALIDATION",
)


if len(train_records) != 1167:

    raise RuntimeError(
        f"Unexpected TRAIN size: "
        f"{len(train_records)}"
    )


if len(val_records) != 250:

    raise RuntimeError(
        f"Unexpected VALIDATION size: "
        f"{len(val_records)}"
    )


train_texts = [
    x["complaint_text"]
    for x in train_records
]

val_texts = [
    x["complaint_text"]
    for x in val_records
]


y_train = np.asarray(
    [
        LABEL_TO_ID[
            x["category"]
        ]
        for x in train_records
    ],
    dtype=np.int64,
)


y_val = np.asarray(
    [
        LABEL_TO_ID[
            x["category"]
        ]
        for x in val_records
    ],
    dtype=np.int64,
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 78)
print(
    "STAGE 12E - FROZEN BERTURK "
    "LINEAR PROBE"
)
print("=" * 78)

print(
    f"Device             : "
    f"{device}"
)

if device.type == "cuda":

    print(
        f"GPU                : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM               : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


print(
    f"PyTorch            : "
    f"{torch.__version__}"
)

print(
    f"TRAIN              : "
    f"{len(train_records)}"
)

print(
    f"VALIDATION         : "
    f"{len(val_records)}"
)

print(
    f"MAX_LENGTH         : "
    f"{MAX_LENGTH}"
)

print(
    f"BATCH_SIZE         : "
    f"{BATCH_SIZE}"
)

print(
    "Frozen TEST used   : False"
)


# ============================================================
# LOAD PRETRAINED BASE BERTURK
# ============================================================

print()
print(
    f"Loading tokenizer: "
    f"{MODEL_NAME}"
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME
    )
)


print(
    f"Loading pretrained encoder: "
    f"{MODEL_NAME}"
)


model = (
    AutoModel
    .from_pretrained(
        MODEL_NAME
    )
)


model.to(
    device
)

model.eval()


# Freeze every BERTurk parameter explicitly
for parameter in model.parameters():

    parameter.requires_grad = False


trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


if trainable_parameters != 0:

    raise RuntimeError(
        "Encoder is not fully frozen."
    )


print(
    f"Encoder params     : "
    f"{total_parameters:,}"
)

print(
    f"Trainable encoder  : "
    f"{trainable_parameters:,}"
)


# ============================================================
# EXTRACT FIXED EMBEDDINGS
# ============================================================

print()
print(
    "Extracting TRAIN embeddings..."
)


X_train, train_embedding_seconds = (
    extract_embeddings(
        train_texts,
        tokenizer,
        model,
        device,
        "TRAIN",
    )
)


print()
print(
    "Extracting VALIDATION embeddings..."
)


X_val, val_embedding_seconds = (
    extract_embeddings(
        val_texts,
        tokenizer,
        model,
        device,
        "VALIDATION",
    )
)


if X_train.shape != (
    len(train_records),
    768,
):

    raise RuntimeError(
        f"Unexpected TRAIN embedding shape: "
        f"{X_train.shape}"
    )


if X_val.shape != (
    len(val_records),
    768,
):

    raise RuntimeError(
        f"Unexpected VALIDATION embedding shape: "
        f"{X_val.shape}"
    )


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    TRAIN_EMBEDDINGS_OUTPUT,
    X_train,
)

np.save(
    VAL_EMBEDDINGS_OUTPUT,
    X_val,
)


# Free GPU memory - sklearn works on CPU
del model

if device.type == "cuda":

    torch.cuda.empty_cache()


# ============================================================
# LINEAR PROBE
#
# Same C candidates used in the classical baseline family.
#
# Selection:
# 1) higher validation Macro-F1
# 2) lower validation log loss
# 3) higher validation accuracy
# ============================================================

candidate_results = []

best_pipeline = None
best_result = None


print()
print(
    "Training Logistic Regression probes..."
)


for C in C_VALUES:

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                C=C,
                max_iter=3000,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


    fit_start = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - fit_start
    )


    train_pred = pipeline.predict(
        X_train
    )

    train_prob = pipeline.predict_proba(
        X_train
    )


    val_pred = pipeline.predict(
        X_val
    )

    val_prob = pipeline.predict_proba(
        X_val
    )


    train_metrics = calculate_metrics(
        y_train,
        train_pred,
        train_prob,
    )

    val_metrics = calculate_metrics(
        y_val,
        val_pred,
        val_prob,
    )


    result = {
        "C":
            C,

        "fit_seconds":
            fit_seconds,

        "train_accuracy":
            train_metrics[
                "accuracy"
            ],

        "train_macro_f1":
            train_metrics[
                "macro_f1"
            ],

        "train_log_loss":
            train_metrics[
                "log_loss"
            ],

        "validation_accuracy":
            val_metrics[
                "accuracy"
            ],

        "validation_macro_precision":
            val_metrics[
                "macro_precision"
            ],

        "validation_macro_recall":
            val_metrics[
                "macro_recall"
            ],

        "validation_macro_f1":
            val_metrics[
                "macro_f1"
            ],

        "validation_weighted_f1":
            val_metrics[
                "weighted_f1"
            ],

        "validation_log_loss":
            val_metrics[
                "log_loss"
            ],
    }


    candidate_results.append(
        result
    )


    print(
        f"C={C:<4} | "
        f"train F1="
        f"{train_metrics['macro_f1']:.6f} | "
        f"val F1="
        f"{val_metrics['macro_f1']:.6f} | "
        f"val acc="
        f"{val_metrics['accuracy']:.6f} | "
        f"val loss="
        f"{val_metrics['log_loss']:.6f}"
    )


    if best_result is None:

        best_result = result
        best_pipeline = pipeline

    else:

        current_key = (
            result[
                "validation_macro_f1"
            ],
            -result[
                "validation_log_loss"
            ],
            result[
                "validation_accuracy"
            ],
        )

        best_key = (
            best_result[
                "validation_macro_f1"
            ],
            -best_result[
                "validation_log_loss"
            ],
            best_result[
                "validation_accuracy"
            ],
        )

        if current_key > best_key:

            best_result = result
            best_pipeline = pipeline


# ============================================================
# SAVE CANDIDATE CSV
# ============================================================

import csv


with CANDIDATES_OUTPUT.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            candidate_results[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        candidate_results
    )


# ============================================================
# SAVE SELECTED PROBE
# ============================================================

joblib.dump(
    {
        "pipeline":
            best_pipeline,

        "model_name":
            MODEL_NAME,

        "representation":
            "BERT pooler_output",

        "encoder_frozen":
            True,

        "max_length":
            MAX_LENGTH,

        "label_to_id":
            LABEL_TO_ID,

        "id_to_label":
            {
                i: label
                for label, i
                in LABEL_TO_ID.items()
            },

        "selected_C":
            best_result["C"],

        "random_seed":
            RANDOM_SEED,

        "train_sha256":
            train_hash,

        "validation_sha256":
            val_hash,

        "test_accessed":
            False,
    },
    MODEL_OUTPUT,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "status":
        "FROZEN_BERTURK_LINEAR_PROBE_COMPLETE",

    "experiment_type":
        "frozen pretrained encoder + linear probe",

    "base_model":
        MODEL_NAME,

    "representation":
        "pooler_output",

    "encoder_frozen":
        True,

    "encoder_trainable_parameters":
        trainable_parameters,

    "encoder_total_parameters":
        total_parameters,

    "device":
        str(device),

    "gpu_name":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),

    "pytorch_version":
        torch.__version__,

    "max_length":
        MAX_LENGTH,

    "batch_size":
        BATCH_SIZE,

    "train_size":
        len(train_records),

    "validation_size":
        len(val_records),

    "train_sha256":
        train_hash,

    "validation_sha256":
        val_hash,

    "train_embedding_seconds":
        train_embedding_seconds,

    "validation_embedding_seconds":
        val_embedding_seconds,

    "candidate_C_values":
        C_VALUES,

    "selection_metric":
        "validation_macro_f1",

    "tie_breakers": [
        "lower_validation_log_loss",
        "higher_validation_accuracy",
    ],

    "selected_C":
        best_result["C"],

    "selected_metrics":
        best_result,

    "test_accessed":
        False,

    "note":
        (
            "This experiment measures how well "
            "the pretrained BERTurk representation "
            "supports the municipal classification "
            "task without updating encoder weights. "
            "It is a control experiment, not the "
            "production classifier."
        ),
}


with SUMMARY_OUTPUT.open(
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
# FINAL OUTPUT
# ============================================================

print()
print("=" * 78)
print(
    "FROZEN BERTURK RESULT"
)
print("=" * 78)

print(
    f"Selected C         : "
    f"{best_result['C']}"
)

print(
    f"Train accuracy     : "
    f"{best_result['train_accuracy']:.6f}"
)

print(
    f"Train Macro-F1     : "
    f"{best_result['train_macro_f1']:.6f}"
)

print(
    f"Validation accuracy: "
    f"{best_result['validation_accuracy']:.6f}"
)

print(
    f"Validation Macro-P : "
    f"{best_result['validation_macro_precision']:.6f}"
)

print(
    f"Validation Macro-R : "
    f"{best_result['validation_macro_recall']:.6f}"
)

print(
    f"Validation Macro-F1: "
    f"{best_result['validation_macro_f1']:.6f}"
)

print(
    f"Validation W-F1    : "
    f"{best_result['validation_weighted_f1']:.6f}"
)

print(
    f"Validation log loss: "
    f"{best_result['validation_log_loss']:.6f}"
)

print()
print(
    f"TRAIN embedding sec: "
    f"{train_embedding_seconds:.2f}"
)

print(
    f"VAL embedding sec  : "
    f"{val_embedding_seconds:.2f}"
)

print()
print(
    f"Saved probe        : "
    f"{MODEL_OUTPUT}"
)

print(
    f"Summary            : "
    f"{SUMMARY_OUTPUT}"
)

print(
    f"Candidates         : "
    f"{CANDIDATES_OUTPUT}"
)

print()
print(
    "Frozen TEST used   : False"
)

print()
print(
    "STATUS: "
    "FROZEN_BERTURK_LINEAR_PROBE_COMPLETE"
)