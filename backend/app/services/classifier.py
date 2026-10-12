import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from backend.app.core.config import settings


LABELS = [
    "TEMIZLIK_ATIK",
    "YOL_KALDIRIM",
    "PARK_YESIL_ALAN",
    "KENT_DUZENI_DENETIM",
    "HAYVAN_VETERINER",
]


class BerturkClassifier:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = None
        self.model = None

        self.top1_threshold: float | None = None
        self.margin_threshold: float | None = None

        self.loaded = False

    def load(self) -> None:
        model_dir = Path(
            settings.model_dir
        )

        policy_path = Path(
            settings.review_policy_path
        )

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: "
                f"{model_dir}"
            )

        if not policy_path.exists():
            raise FileNotFoundError(
                f"Review policy not found: "
                f"{policy_path}"
            )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_dir,
                local_files_only=True,
            )
        )

        self._validate_model_labels()

        self.model.to(
            self.device
        )

        self.model.eval()

        with policy_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            policy = json.load(f)

        self.top1_threshold = float(
            policy["top1_threshold"]
        )

        self.margin_threshold = float(
            policy["margin_threshold"]
        )

        self.loaded = True

    def _validate_model_labels(self) -> None:
        raw_id2label = (
            self.model.config.id2label
        )

        normalized = {
            int(index): label
            for index, label
            in raw_id2label.items()
        }

        expected = {
            index: label
            for index, label
            in enumerate(LABELS)
        }

        if normalized != expected:
            raise RuntimeError(
                "Model label mapping does not "
                "match project label order.\n"
                f"Expected: {expected}\n"
                f"Found   : {normalized}"
            )

    def _review_decision(
        self,
        top1_score: float,
        margin: float,
    ) -> tuple[bool, str | None]:

        low_confidence = (
            top1_score
            < self.top1_threshold
        )

        close_top2 = (
            margin
            < self.margin_threshold
        )

        if (
            low_confidence
            and close_top2
        ):
            return (
                True,
                "LOW_CONFIDENCE_AND_CLOSE_TOP2",
            )

        if low_confidence:
            return (
                True,
                "LOW_CONFIDENCE",
            )

        if close_top2:
            return (
                True,
                "CLOSE_TOP2",
            )

        return (
            False,
            None,
        )

    def predict(
        self,
        text: str,
    ) -> dict:

        if not self.loaded:
            raise RuntimeError(
                "Classifier has not been loaded."
            )

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError(
                "Complaint text cannot be empty."
            )

        encoded = self.tokenizer(
            cleaned_text,
            truncation=True,
            max_length=settings.model_max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():
            outputs = self.model(
                **encoded
            )

            probabilities = (
                torch.softmax(
                    outputs.logits,
                    dim=-1,
                )[0]
                .detach()
                .cpu()
            )

        order = torch.argsort(
            probabilities,
            descending=True,
        )

        top1_id = int(
            order[0].item()
        )

        top2_id = int(
            order[1].item()
        )

        top1_score = float(
            probabilities[
                top1_id
            ].item()
        )

        top2_score = float(
            probabilities[
                top2_id
            ].item()
        )

        margin = (
            top1_score
            - top2_score
        )

        (
            review_required,
            review_reason,
        ) = self._review_decision(
            top1_score,
            margin,
        )

        return {
            "predicted_category":
                LABELS[top1_id],

            "secondary_category":
                LABELS[top2_id],

            "model_score":
                top1_score,

            "second_model_score":
                top2_score,

            "score_margin":
                margin,

            "review_required":
                review_required,

            "review_reason":
                review_reason,

            "model_version":
                settings.model_version,

            "device":
                str(self.device),
        }