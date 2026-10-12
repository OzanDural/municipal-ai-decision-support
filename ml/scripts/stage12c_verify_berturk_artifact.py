from pathlib import Path
import hashlib
import json

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_DIR = Path(
    "ml/models/berturk/final_model"
)

MODEL_WEIGHTS = (
    MODEL_DIR
    / "model.safetensors"
)

REPORT_DIR = Path(
    "ml/reports/model_training"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "12c_berturk_artifact_verification.json"
)


REQUIRED_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_history.csv",
    "training_summary.json",
    "validation_metrics.json",
    "epoch_metrics.csv",
    "validation_predictions.csv",
}


EXPECTED_MODEL_SHA256 = (
    "af3aaf2d03972e0b07f5858971b9114b"
    "efea21b7127e390d088e3c85e7bd1bb4"
)


EXPECTED_LABELS = {
    "TEMIZLIK_ATIK",
    "YOL_KALDIRIM",
    "PARK_YESIL_ALAN",
    "KENT_DUZENI_DENETIM",
    "HAYVAN_VETERINER",
}


def sha256_file(
    path: Path,
) -> str:
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
# VERIFY CANONICAL MODEL ARTIFACT
# ============================================================

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Model directory not found:\n"
        f"{MODEL_DIR}"
    )


missing_files = {
    name
    for name in REQUIRED_FILES
    if not (
        MODEL_DIR / name
    ).exists()
}

if missing_files:
    raise RuntimeError(
        "Required model files are missing:\n"
        + "\n".join(
            sorted(missing_files)
        )
    )


model_hash = sha256_file(
    MODEL_WEIGHTS
)

model_size_mb = (
    MODEL_WEIGHTS.stat().st_size
    / 1024**2
)


if (
    model_hash
    != EXPECTED_MODEL_SHA256
):
    raise RuntimeError(
        "Fine-tuned BERTurk model "
        "SHA256 mismatch.\n"
        f"Expected: "
        f"{EXPECTED_MODEL_SHA256}\n"
        f"Found   : "
        f"{model_hash}"
    )


MODEL_ROOT = MODEL_DIR


# ============================================================
# LOAD TOKENIZER + MODEL
# ============================================================

print(
    "Tokenizer yükleniyor..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
    )
)


print(
    "BERTurk modeli CPU üzerinde "
    "yükleniyor..."
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
    )
)

model.eval()


# ============================================================
# VERIFY MODEL CONFIG
# ============================================================

config_labels = set(
    model.config.id2label.values()
)


if config_labels != EXPECTED_LABELS:
    raise RuntimeError(
        "Model label mapping beklenen "
        "5 sınıfla eşleşmiyor.\n"
        f"Model: {config_labels}"
    )


if model.config.num_labels != 5:
    raise RuntimeError(
        "Model num_labels != 5"
    )


# ============================================================
# LOCAL INFERENCE SANITY CHECK
# ============================================================

sanity_text = (
    "Mahalledeki çöp konteyneri "
    "tamamen dolmuş ve çöpler "
    "çevreye taşmış durumda."
)


inputs = tokenizer(
    sanity_text,
    return_tensors="pt",
    truncation=True,
    max_length=160,
)


with torch.no_grad():
    outputs = model(
        **inputs
    )

    probabilities = (
        torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]
    )


predicted_id = int(
    probabilities.argmax()
)

predicted_label = (
    model.config.id2label[
        predicted_id
    ]
)

top_score = float(
    probabilities[
        predicted_id
    ]
)


if (
    predicted_label
    not in EXPECTED_LABELS
):
    raise RuntimeError(
        "Inference invalid label döndürdü."
    )


# ============================================================
# TRAINING SUMMARY
# ============================================================

training_summary_path = (
    MODEL_ROOT
    / "training_summary.json"
)


with training_summary_path.open(
    "r",
    encoding="utf-8",
) as f:
    training_summary = json.load(
        f
    )


# ============================================================
# SAVE VERIFICATION
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


summary = {
    "status":
        "BERTURK_ARTIFACT_VERIFIED",

    "artifact_type":
        "canonical_model_directory",

    "model_root":
        str(MODEL_ROOT),

    "model_weights":
        str(MODEL_WEIGHTS),

    "model_size_mb":
        round(
            model_size_mb,
            3,
        ),

    "model_sha256":
        model_hash,

    "required_files_present":
        True,

    "model_class":
        model.__class__.__name__,

    "num_labels":
        model.config.num_labels,

    "id2label": {
        str(k): v
        for k, v
        in model.config.id2label.items()
    },

    "sanity_inference": {
        "text":
            sanity_text,

        "predicted_category":
            predicted_label,

        "top1_score":
            top_score,
    },

    "training_summary":
        training_summary,

    "frozen_test_accessed":
        False,
}


with SUMMARY_PATH.open(
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

print("\n" + "=" * 72)
print(
    "STAGE 12C - BERTURK ARTIFACT "
    "VERIFICATION"
)
print("=" * 72)

print(
    f"Model size MB      : "
    f"{model_size_mb:.2f}"
)

print(
    f"Model SHA-256      : "
    f"{model_hash}"
)

print(
    "Required files     : PASS"
)

print(
    f"Model              : "
    f"{model.__class__.__name__}"
)

print(
    f"Number of labels   : "
    f"{model.config.num_labels}"
)

print(
    "Label mapping      : PASS"
)

print(
    f"Sanity prediction  : "
    f"{predicted_label}"
)

print(
    f"Sanity top-1 score : "
    f"{top_score:.6f}"
)

print(
    "Frozen TEST used   : False"
)

print()
print(
    f"Verification report: "
    f"{SUMMARY_PATH}"
)

print()
print(
    "STATUS: "
    "BERTURK_ARTIFACT_VERIFIED"
)
