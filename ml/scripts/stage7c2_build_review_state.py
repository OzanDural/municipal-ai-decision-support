import csv
import json
from collections import Counter
from pathlib import Path


MASTER_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

DECISIONS_FILE = Path(
    "ml/reports/dataset_qa/"
    "07c_provisional_duplicate_decisions.csv"
)

RESIDUAL_FILE = Path(
    "ml/reports/dataset_qa/"
    "07b_residual_pair_review.csv"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

LEDGER_FILE = REPORT_DIR / (
    "07c_review_ledger.csv"
)

SURVIVOR_FILE = REPORT_DIR / (
    "07c_survivor_view.jsonl"
)

FILTERED_RESIDUAL_FILE = REPORT_DIR / (
    "07c_residual_pairs_after_provisional.csv"
)

SUMMARY_FILE = REPORT_DIR / (
    "07c2_review_state_summary.json"
)


# ==================================================
# Master dataset
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
        f"Beklenen 1750 kayıt, bulunan {len(records)}."
    )


record_ids = {
    r["sample_id"]
    for r in records
}


# ==================================================
# Review decisions
# ==================================================

decision_rows = []

with DECISIONS_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        decision_rows.append(row)


# Aynı sample_id birden fazla satırda bulunmuşsa
# çelişki var mı kontrol et.
decision_map = {}


for row in decision_rows:

    sample_id = row["sample_id"]

    if sample_id not in record_ids:
        raise ValueError(
            f"Decision dosyasında dataset'te bulunmayan "
            f"sample_id var: {sample_id}"
        )

    decision = row["decision"]

    if sample_id in decision_map:

        previous = decision_map[sample_id]

        if previous["decision"] != decision:
            raise ValueError(
                f"Çelişkili karar: {sample_id} "
                f"{previous['decision']} vs {decision}"
            )

        # Aynı karar tekrar geldiyse ilk satırı koru.
        continue

    decision_map[sample_id] = row


reject_ids = {
    sample_id
    for sample_id, row
    in decision_map.items()
    if row["decision"] == "REJECT_DUPLICATE"
}


keep_ids = {
    sample_id
    for sample_id, row
    in decision_map.items()
    if row["decision"] == "KEEP"
}


if reject_ids & keep_ids:
    raise ValueError(
        "Aynı kayıt hem KEEP hem REJECT olmuş."
    )


# ==================================================
# Review ledger
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


with LEDGER_FILE.open(
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

        row = decision_map[
            sample_id
        ]

        writer.writerow({
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
                "STAGE_7C_DUPLICATE_REVIEW",

            # Bunlar hâlâ provisional.
            "finalized":
                False,
        })


# ==================================================
# Survivor view
#
# Bu FINAL DATASET DEĞİLDİR.
# Sadece sonraki QA aşamalarında kullanılan
# geçici analiz görünümüdür.
# ==================================================

survivors = [
    record
    for record in records
    if record["sample_id"]
    not in reject_ids
]


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
# Residual pairs yeniden filtrele
# ==================================================

residual_rows = []

with RESIDUAL_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    residual_fields = (
        reader.fieldnames
    )

    for row in reader:
        residual_rows.append(row)


filtered_residual = []

removed_residual = []


for row in residual_rows:

    id_1 = row["sample_id_1"]
    id_2 = row["sample_id_2"]

    if (
        id_1 in reject_ids
        or id_2 in reject_ids
    ):

        removed_residual.append(
            row
        )

        continue

    filtered_residual.append(
        row
    )


with FILTERED_RESIDUAL_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=residual_fields,
    )

    writer.writeheader()

    writer.writerows(
        filtered_residual
    )


# ==================================================
# Checks
# ==================================================

expected_survivor_count = (
    len(records)
    - len(reject_ids)
)


if (
    len(survivors)
    != expected_survivor_count
):
    raise ValueError(
        "Survivor count kontrolü başarısız."
    )


# kept_reference reject edilmemiş olmalı.
bad_references = []


for sample_id, row in (
    decision_map.items()
):

    if row["decision"] != (
        "REJECT_DUPLICATE"
    ):
        continue

    reference = (
        row["kept_reference"]
    )

    if not reference:
        bad_references.append({
            "sample_id":
                sample_id,

            "problem":
                "EMPTY_REFERENCE",
        })

    elif reference in reject_ids:

        bad_references.append({
            "sample_id":
                sample_id,

            "problem":
                "REFERENCE_IS_REJECTED",

            "reference":
                reference,
        })

    elif reference not in record_ids:

        bad_references.append({
            "sample_id":
                sample_id,

            "problem":
                "REFERENCE_NOT_FOUND",

            "reference":
                reference,
        })


if bad_references:
    raise ValueError(
        "kept_reference kontrolü başarısız:\n"
        + json.dumps(
            bad_references,
            ensure_ascii=False,
            indent=2,
        )
    )


# ==================================================
# Summary
# ==================================================

decision_counts = Counter(
    row["decision"]
    for row
    in decision_map.values()
)

reason_counts = Counter(
    row["reason_code"]
    for row
    in decision_map.values()
)


residual_reason_counts = Counter(
    row["residual_reason"]
    for row
    in filtered_residual
)


summary = {
    "master_record_count":
        len(records),

    "unique_reviewed_record_count":
        len(decision_map),

    "provisional_keep_count":
        len(keep_ids),

    "provisional_reject_count":
        len(reject_ids),

    "survivor_view_record_count":
        len(survivors),

    "original_residual_pair_count":
        len(residual_rows),

    "removed_residual_pair_count":
        len(removed_residual),

    "remaining_residual_pair_count":
        len(filtered_residual),

    "decision_counts":
        dict(decision_counts),

    "reason_counts":
        dict(reason_counts),

    "remaining_residual_reason_counts":
        dict(
            residual_reason_counts
        ),

    "master_dataset_modified":
        False,

    "survivor_view_is_final_dataset":
        False,

    "policy":
        (
            "Provisional duplicate-review decisions "
            "are stored in a separate audit ledger. "
            "master_stage2.jsonl remains immutable. "
            "The survivor view is used only to simplify "
            "subsequent semantic and rule-based review."
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
print("STAGE 7C-2 - REVIEW STATE")
print("=" * 78)

print(
    f"Master kayıt          : {len(records)}"
)

print(
    f"Review edilen unique  : {len(decision_map)}"
)

print(
    f"Provisional KEEP      : {len(keep_ids)}"
)

print(
    f"Provisional REJECT    : {len(reject_ids)}"
)

print(
    f"Survivor view         : {len(survivors)}"
)

print("\nResidual pairs:")

print(
    f"  Önce                : {len(residual_rows)}"
)

print(
    f"  Elenen               : {len(removed_residual)}"
)

print(
    f"  Kalan                : {len(filtered_residual)}"
)

print("\nKalan residual nedenleri:")

for key, value in sorted(
    residual_reason_counts.items()
):

    print(
        f"  {key:<40}: {value}"
    )


print("\nDosyalar:")

print(
    f"  Ledger   : {LEDGER_FILE}"
)

print(
    f"  Survivor : {SURVIVOR_FILE}"
)

print(
    f"  Residual : {FILTERED_RESIDUAL_FILE}"
)

print(
    f"  Summary  : {SUMMARY_FILE}"
)