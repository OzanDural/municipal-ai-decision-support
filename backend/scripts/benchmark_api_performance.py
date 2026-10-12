import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


BASE_URL = "http://127.0.0.1:8000"

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
            percentile * len(ordered)
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
    with httpx.Client(
        base_url=BASE_URL,
        timeout=30.0,
    ) as client:

        # -----------------------------------
        # Güvenlik kontrolü
        # -----------------------------------

        health_response = client.get(
            "/health"
        )

        health_response.raise_for_status()

        health = health_response.json()

        database_name = (
            health["database"]["name"]
        )

        if (
            database_name
            != "belediye_ai_test"
        ):
            raise RuntimeError(
                "SAFETY STOP: Benchmark must "
                "use belediye_ai_test, but "
                f"server is using {database_name}."
            )

        if (
            health["model"]["version"]
            != "berturk_final_v2_99ea9e29"
        ):
            raise RuntimeError(
                "Unexpected model version."
            )

        print(
            "Database:",
            database_name,
        )

        print(
            "Model:",
            health["model"]["version"],
        )

        print(
            "Device:",
            health["model"]["device"],
        )

        # -----------------------------------
        # Warm-up
        # -----------------------------------

        print()
        print(
            f"Running {WARMUP_RUNS} "
            "HTTP warm-up requests..."
        )

        for index in range(
            WARMUP_RUNS
        ):
            response = client.post(
                "/complaints",
                json={
                    "complaint_text":
                        SAMPLE_TEXTS[
                            index
                            % len(SAMPLE_TEXTS)
                        ],

                    "latitude": 37.1110882,
                    
                    "longitude": 35.473158069576726,
                },
            )

            if response.status_code != 201:
                raise RuntimeError(
                    "Warm-up complaint failed: "
                    f"{response.status_code} "
                    f"{response.text}"
                )

        # -----------------------------------
        # Gerçek ölçümler
        # -----------------------------------

        print(
            f"Running {MEASURED_RUNS} "
            "measured HTTP requests..."
        )

        latencies_ms = []

        for index in range(
            MEASURED_RUNS
        ):
            payload = {
                "complaint_text":
                    SAMPLE_TEXTS[
                        index
                        % len(SAMPLE_TEXTS)
                    ],

                "latitude":
                    37.1110882
                    + (
                        index
                        * 0.000001
                    ),

                "longitude":
                     35.473158069576726
                     + (
                           index
                           * 0.000001
                         ),
                    
            }

            start = time.perf_counter()

            response = client.post(
                "/complaints",
                json=payload,
            )

            elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000

            if response.status_code != 201:
                raise RuntimeError(
                    "Complaint request failed: "
                    f"{response.status_code} "
                    f"{response.text}"
                )

            data = response.json()

            if (
                data["model_version"]
                != "berturk_final_v2_99ea9e29"
            ):
                raise RuntimeError(
                    "Request did not use "
                    "the real final model."
                )

            latencies_ms.append(
                elapsed_ms
            )

            print(
                f"{index + 1:02d}/"
                f"{MEASURED_RUNS} "
                f"{elapsed_ms:.3f} ms "
                f"{data['predicted_category']}"
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

        "endpoint":
            "POST /complaints",

        "database":
            "belediye_ai_test",

        "model_version":
            "berturk_final_v2_99ea9e29",

        "warmup_runs":
            WARMUP_RUNS,

        "measured_runs":
            MEASURED_RUNS,

        "http_response_latency_ms": {
            "mean": mean_ms,
            "median": median_ms,
            "p95": p95_ms,
            "min": min_ms,
            "max": max_ms,
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
        / "complaint_api_performance_v1.json"
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
    print(
        "REAL COMPLAINT API PERFORMANCE"
    )
    print("=" * 60)

    print(
        f"Mean response       : "
        f"{mean_ms:.3f} ms"
    )

    print(
        f"Median response     : "
        f"{median_ms:.3f} ms"
    )

    print(
        f"P95 response        : "
        f"{p95_ms:.3f} ms"
    )

    print(
        f"Minimum response    : "
        f"{min_ms:.3f} ms"
    )

    print(
        f"Maximum response    : "
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
        "REAL_COMPLAINT_API_BENCHMARK_OK"
    )


if __name__ == "__main__":
    main()