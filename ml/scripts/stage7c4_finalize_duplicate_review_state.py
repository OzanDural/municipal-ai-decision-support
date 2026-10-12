import csv
import json
from collections import Counter
from pathlib import Path


MASTER_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

OLD_LEDGER_FILE = Path(
    "ml/reports/dataset_qa/"
    "07c_review_ledger.csv"
)

NEW_DECISIONS_FILE = Path(
    "ml/reports/dataset_qa/"
    "07c3_residual_duplicate_decisions.csv"
)

PAIR_AUDIT_FILE = Path(
    "ml/reports/dataset_qa/"
    "07c3_residual_pair_audit.csv"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

FINAL_LEDGER_FILE = REPORT_DIR / (
    "07c_duplicate_review_ledger.csv"
)

SURVIVOR_FILE = REPORT_DIR / (
    "07c_duplicate_review_survivor.jsonl"
)

SUMMARY_FILE = REPORT_DIR / (
    "07c_duplicate_review_summary.json"
)


# ==================================================
# Master
# ==================================================

records = []

with MASTER_FILE.open(
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


if len(records) != 1750:

    raise ValueError(
        f"Beklenen 1750 kayıt, "
        f"bulunan {len(records)}."
    )


record_ids = {
    r["sample_id"]
    for r in records
}


# ==================================================
# Existing ledger
# ==================================================

decision_map = {}


with OLD_LEDGER_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        sample_id = row[
            "sample_id"
        ]

        decision_map[
            sample_id
        ] = {
            "sample_id":
                sample_id,

            "decision":
                row["decision"],

            "reason_code":
                row["reason_code"],

            "kept_reference":
                row["kept_reference"],

            "review_scope":
                row["review_scope"],

            "group_id":
                row["group_id"],

            "review_note":
                row["review_note"],

            "review_stage":
                row["review_stage"],

            "finalized":
                False,
        }


# ==================================================
# New Stage 7C-3 decisions
# ==================================================

with NEW_DECISIONS_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        sample_id = row[
            "sample_id"
        ]

        if sample_id not in record_ids:

            raise ValueError(
                f"Dataset'te bulunmayan sample_id: "
                f"{sample_id}"
            )


        if sample_id in decision_map:

            old_decision = (
                decision_map[
                    sample_id
                ][
                    "decision"
                ]
            )

            if (
                old_decision
                != row["decision"]
            ):

                raise ValueError(
                    f"Çelişkili karar: "
                    f"{sample_id}: "
                    f"{old_decision} vs "
                    f"{row['decision']}"
                )

            # Aynı karar zaten varsa
            # mevcut audit satırını koru.
            continue


        decision_map[
            sample_id
        ] = {
            "sample_id":
                sample_id,

            "decision":
                row["decision"],

            "reason_code":
                row["reason_code"],

            "kept_reference":
                row["kept_reference"],

            "review_scope":
                row["review_scope"],

            "group_id":
                row["component_id"],

            "review_note":
                row["review_note"],

            "review_stage":
                row["review_stage"],

            "finalized":
                False,
        }


# ==================================================
# Decision sets
# ==================================================

reject_ids = {
    sample_id
    for sample_id, row
    in decision_map.items()
    if row["decision"]
    == "REJECT_DUPLICATE"
}


keep_ids = {
    sample_id
    for sample_id, row
    in decision_map.items()
    if row["decision"]
    == "KEEP"
}


if reject_ids & keep_ids:

    raise ValueError(
        "Aynı kayıt hem KEEP "
        "hem REJECT durumunda."
    )


if len(reject_ids) != 58:

    raise ValueError(
        f"Beklenen 58 duplicate reject, "
        f"bulunan {len(reject_ids)}."
    )


# ==================================================
# kept_reference integrity
# ==================================================

reference_errors = []


for sample_id, row in (
    decision_map.items()
):

    if (
        row["decision"]
        != "REJECT_DUPLICATE"
    ):

        continue


    reference = (
        row["kept_reference"]
    )


    if not reference:

        reference_errors.append({
            "sample_id":
                sample_id,

            "problem":
                "EMPTY_REFERENCE",
        })

        continue


    if reference not in record_ids:

        reference_errors.append({
            "sample_id":
                sample_id,

            "problem":
                "REFERENCE_NOT_FOUND",

            "reference":
                reference,
        })


    elif reference in reject_ids:

        reference_errors.append({
            "sample_id":
                sample_id,

            "problem":
                "REFERENCE_IS_REJECTED",

            "reference":
                reference,
        })


if reference_errors:

    raise ValueError(
        "kept_reference integrity "
        "kontrolü başarısız:\n"
        + json.dumps(
            reference_errors,
            ensure_ascii=False,
            indent=2,
        )
    )


# ==================================================
# Pair audit
#
# Stage 7C-3 sonunda bütün residual
# edge'ler review edilmiş olmalı.
# ==================================================

pair_rows = []

with PAIR_AUDIT_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        pair_rows.append(row)


if len(pair_rows) != 74:

    raise ValueError(
        f"Beklenen 74 residual pair, "
        f"bulunan {len(pair_rows)}."
    )


allowed_pair_decisions = {
    "RESOLVED_BY_REJECT",
    "KEEP_BOTH_AFTER_REVIEW",
}


invalid_pairs = [
    row
    for row in pair_rows
    if row[
        "pair_disposition"
    ]
    not in allowed_pair_decisions
]


if invalid_pairs:

    raise ValueError(
        "İncelenmemiş veya geçersiz "
        "pair disposition bulundu."
    )


pair_counts = Counter(
    row[
        "pair_disposition"
    ]
    for row in pair_rows
)


# ==================================================
# Final duplicate-review ledger
# ==================================================

ledger_fields = [
    "sample_id",
    "decision",
    "reason_code",
    "kept_reference",
    "review_scope",
    "group_id",
    "review_note",
    "review_stage",
    "finalized",
]


with FINAL_LEDGER_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=ledger_fields,
    )

    writer.writeheader()


    for sample_id in sorted(
        decision_map
    ):

        writer.writerow(
            decision_map[
                sample_id
            ]
        )


# ==================================================
# Survivor view
#
# HÂLÂ FINAL DATASET DEĞİL.
# Sadece duplicate QA sonrası görünüm.
# ==================================================

survivors = [
    record
    for record in records
    if record[
        "sample_id"
    ]
    not in reject_ids
]


if len(survivors) != 1692:

    raise ValueError(
        f"Beklenen survivor 1692, "
        f"bulunan {len(survivors)}."
    )


with SURVIVOR_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for record in survivors:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ==================================================
# Distribution sanity check
# ==================================================

category_before = Counter(
    r["category"]
    for r in records
)

category_after = Counter(
    r["category"]
    for r in survivors
)

generator_before = Counter(
    r["generator_id"]
    for r in records
)

generator_after = Counter(
    r["generator_id"]
    for r in survivors
)


reject_reason_counts = Counter(
    row["reason_code"]
    for row
    in decision_map.values()
    if row["decision"]
    == "REJECT_DUPLICATE"
)


# ==================================================
# Summary
# ==================================================

summary = {
    "status":
        "DUPLICATE_REVIEW_COMPLETE_PROVISIONAL",

    "master_record_count":
        len(records),

    "master_dataset_modified":
        False,

    "reviewed_unique_record_count":
        len(decision_map),

    "keep_decision_count":
        len(keep_ids),

    "reject_duplicate_count":
        len(reject_ids),

    "duplicate_review_survivor_count":
        len(survivors),

    "stage7c3_residual_pair_count":
        len(pair_rows),

    "residual_pair_dispositions":
        dict(
            pair_counts
        ),

    "unreviewed_residual_pair_count":
        0,

    "category_distribution_before":
        dict(
            category_before
        ),

    "category_distribution_after":
        dict(
            category_after
        ),

    "generator_distribution_before":
        dict(
            generator_before
        ),

    "generator_distribution_after":
        dict(
            generator_after
        ),

    "reject_reason_counts":
        dict(
            reject_reason_counts
        ),

    "duplicate_review_finalized":
        False,

    "survivor_is_final_dataset":
        False,

    "policy": (
        "Duplicate review is complete, but decisions "
        "remain provisional until all other Stage 7 "
        "quality reviews are complete. master_stage2 "
        "remains immutable. No similarity threshold "
        "was used as an automatic deletion rule."
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


# ==================================================
# Terminal
# ==================================================

print("=" * 78)
print("STAGE 7C-4 - DUPLICATE REVIEW STATE")
print("=" * 78)

print(
    f"Master kayıt             : "
    f"{len(records)}"
)

print(
    f"Review edilen unique     : "
    f"{len(decision_map)}"
)

print(
    f"KEEP                     : "
    f"{len(keep_ids)}"
)

print(
    f"REJECT_DUPLICATE         : "
    f"{len(reject_ids)}"
)

print(
    f"Duplicate survivor       : "
    f"{len(survivors)}"
)

print("\nResidual pair audit:")

for key, value in sorted(
    pair_counts.items()
):

    print(
        f"  {key:<30}: {value}"
    )


print(
    "\nUnreviewed residual pair : 0"
)


print("\nKategori dağılımı:")

for category in sorted(
    category_before
):

    print(
        f"  {category:<24} "
        f"{category_before[category]} "
        f"-> "
        f"{category_after[category]}"
    )


print("\nDosyalar:")

print(
    f"  Ledger   : {FINAL_LEDGER_FILE}"
)

print(
    f"  Survivor : {SURVIVOR_FILE}"
)

print(
    f"  Summary  : {SUMMARY_FILE}"
)