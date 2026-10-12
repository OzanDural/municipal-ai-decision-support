import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.app.services.classifier import (
    BerturkClassifier,
)


WARMUP_RUNS = 5
MEASURED_RUNS = 30


SAMPLE_TEXTS = [
    (
        "Sokağımızdaki çöp konteyneri "
        "günlerdir boşaltılmadı ve çöpler "
        "etrafa yayılmaya başladı."
    ),
    (
        "Mahalledeki kaldırım taşları "
        "kırılmış durumda, yürürken "
        "ayağımız takılıyor."
    ),
    (
        "Parktaki çocuk oyun grubunun "
        "salıncağı kırılmış ve kullanılması "
        "tehlikeli hale gelmiş."
    ),
    (
        "İşletme masa ve sandalyelerini "
        "kaldırımın tamamına yerleştirdiği "
        "için yayalar geçemiyor."
    ),
    (
        "Sokakta yaralı halde duran bir "
        "köpek var, hareket etmekte "
        "zorlanıyor ve yardıma ihtiyacı var."
    ),
]


def percentile_nearest_rank(
    values: list[float],
    percentile: float,
) -> float:
    ordered = sorted(values)

    index = (
        math.ceil(
            percentile
            * len(ordered)
        )
        - 1
    )

    index = max(
        0,
        min(
            index,
            len(ordered) - 1,
        ),
    )

    return ordered[index]


def main() -> None:
    classifier = BerturkClassifier()

    print(
        "Loading real fine-tuned BERTurk..."
    )

    load_start = time.perf_counter()

    classifier.load()

    model_load_seconds = (
        time.perf_counter()
        - load_start
    )

    print(
        f"Model loaded in "
        f"{model_load_seconds:.3f} s"
    )

    print(
        f"Device: {classifier.device}"
    )

    print()
    print(
        f"Running {WARMUP_RUNS} "
        "warm-up inferences..."
    )

    for index in range(
        WARMUP_RUNS
    ):
        text = SAMPLE_TEXTS[
            index
            % len(SAMPLE_TEXTS)
        ]

        classifier.predict(
            text
        )

    print(
        f"Running {MEASURED_RUNS} "
        "measured inferences..."
    )

    latencies_ms = []

    for index in range(
        MEASURED_RUNS
    ):
        text = SAMPLE_TEXTS[
            index
            % len(SAMPLE_TEXTS)
        ]

        start = time.perf_counter()

        result = classifier.predict(
            text
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        latencies_ms.append(
            elapsed_ms
        )

        print(
            f"{index + 1:02d}/"
            f"{MEASURED_RUNS} "
            f"{elapsed_ms:.3f} ms "
            f"{result['predicted_category']}"
        )

    mean_ms = statistics.mean(
        latencies_ms
    )

    median_ms = statistics.median(
        latencies_ms
    )

    p95_ms = percentile_nearest_rank(
        latencies_ms,
        0.95,
    )

    min_ms = min(
        latencies_ms
    )

    max_ms = max(
        latencies_ms
    )

    results = {
        "measured_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model_version":
            "berturk_final_v2_99ea9e29",

        "device":
            str(
                classifier.device
            ),

        "model_load_seconds":
            model_load_seconds,

        "warmup_runs":
            WARMUP_RUNS,

        "measured_runs":
            MEASURED_RUNS,

        "inference_latency_ms": {
            "mean":
                mean_ms,

            "median":
                median_ms,

            "p95":
                p95_ms,

            "min":
                min_ms,

            "max":
                max_ms,
        },
    }

    output_dir = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "benchmark_results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "model_performance_v1.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 60)
    print("BERTURK PERFORMANCE RESULTS")
    print("=" * 60)

    print(
        f"Device             : "
        f"{classifier.device}"
    )

    print(
        f"Model load         : "
        f"{model_load_seconds:.3f} s"
    )

    print(
        f"Mean inference     : "
        f"{mean_ms:.3f} ms"
    )

    print(
        f"Median inference   : "
        f"{median_ms:.3f} ms"
    )

    print(
        f"P95 inference      : "
        f"{p95_ms:.3f} ms"
    )

    print(
        f"Minimum inference  : "
        f"{min_ms:.3f} ms"
    )

    print(
        f"Maximum inference  : "
        f"{max_ms:.3f} ms"
    )

    print()
    print(
        f"Saved to: "
        f"{output_path}"
    )

    print()
    print(
        "STATUS: "
        "REAL_BERTURK_BENCHMARK_OK"
    )


if __name__ == "__main__":
    main()