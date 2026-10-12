import time

from backend.app.services.classifier import (
    BerturkClassifier,
)


def main() -> None:
    classifier = BerturkClassifier()

    load_start = time.perf_counter()

    classifier.load()

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    test_text = (
        "Mahallemizdeki çöp konteyneri "
        "günlerdir boşaltılmadı ve çöpler "
        "sokağa taşmış durumda."
    )

    inference_start = (
        time.perf_counter()
    )

    result = classifier.predict(
        test_text
    )

    inference_ms = (
        (
            time.perf_counter()
            - inference_start
        )
        * 1000
    )

    print("=" * 70)
    print(
        "STAGE 16A - BERTURK "
        "INFERENCE SERVICE TEST"
    )
    print("=" * 70)

    print(
        f"Device             : "
        f"{result['device']}"
    )

    print(
        f"Model load seconds : "
        f"{load_seconds:.3f}"
    )

    print(
        f"Inference ms       : "
        f"{inference_ms:.3f}"
    )

    print()
    print(
        f"Prediction         : "
        f"{result['predicted_category']}"
    )

    print(
        f"Secondary          : "
        f"{result['secondary_category']}"
    )

    print(
        f"Top-1 score        : "
        f"{result['model_score']:.6f}"
    )

    print(
        f"Top-2 score        : "
        f"{result['second_model_score']:.6f}"
    )

    print(
        f"Margin             : "
        f"{result['score_margin']:.6f}"
    )

    print(
        f"Review required    : "
        f"{result['review_required']}"
    )

    print(
        f"Review reason      : "
        f"{result['review_reason']}"
    )

    print(
        f"Model version      : "
        f"{result['model_version']}"
    )

    print()
    print(
        "STATUS: "
        "BERTURK_INFERENCE_SERVICE_OK"
    )


if __name__ == "__main__":
    main()