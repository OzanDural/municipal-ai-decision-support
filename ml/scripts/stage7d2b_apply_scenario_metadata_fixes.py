import csv
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path


SURVIVOR_INPUT = Path(
    "ml/reports/dataset_qa/"
    "07d1b_survivor_after_artificial.jsonl"
)

PREVIOUS_STAGE_LEDGER = Path(
    "ml/reports/dataset_qa/"
    "07d1b_stage_review_ledger.csv"
)

CORRECTIONS_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d2_scenario_metadata_corrections.csv"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

SURVIVOR_OUTPUT = REPORT_DIR / (
    "07d2b_survivor_after_scenario_fix.jsonl"
)

STAGE_LEDGER_OUTPUT = REPORT_DIR / (
    "07d2b_stage_review_ledger.csv"
)

APPLIED_AUDIT_FILE = REPORT_DIR / (
    "07d2b_scenario_fix_audit.csv"
)

SUMMARY_FILE = REPORT_DIR / (
    "07d2b_scenario_fix_summary.json"
)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def normalize_bool(value):
    value = str(value).strip().lower()

    if value in {"true", "1", "yes"}:
        return True

    if value in {"false", "0", "no", ""}:
        return False

    raise ValueError(
        f"Geçersiz boolean değer: {value}"
    )


# ==================================================
# Survivor input
# ==================================================

records = []

with SURVIVOR_INPUT.open(
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


if len(records) != 1681:
    raise ValueError(
        f"Beklenen survivor 1681, "
        f"bulunan {len(records)}."
    )


record_map = {
    r["sample_id"]: r
    for r in records
}


if len(record_map) != 1681:
    raise ValueError(
        "Survivor içinde duplicate sample_id var."
    )


# ==================================================
# Corrections
# ==================================================

corrections = []

with CORRECTIONS_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    required_fields = {
        "sample_id",
        "field",
        "old_value",
        "new_value",
        "decision",
        "reason_code",
        "review_note",
        "review_stage",
        "finalized",
    }

    if not reader.fieldnames:
        raise ValueError(
            "Correction CSV header bulunamadı."
        )

    missing = (
        required_fields
        - set(reader.fieldnames)
    )

    if missing:
        raise ValueError(
            f"Eksik correction kolonları: "
            f"{sorted(missing)}"
        )

    for row in reader:
        corrections.append(row)


if len(corrections) != 4:
    raise ValueError(
        f"Beklenen 4 metadata correction, "
        f"bulunan {len(corrections)}."
    )


correction_ids = [
    row["sample_id"]
    for row in corrections
]


if len(set(correction_ids)) != 4:
    raise ValueError(
        "Correction dosyasında duplicate sample_id var."
    )


# ==================================================
# Apply corrections
# ==================================================

corrected_records = [
    deepcopy(r)
    for r in records
]

corrected_map = {
    r["sample_id"]: r
    for r in corrected_records
}


audit_rows = []


for row in corrections:

    sample_id = row["sample_id"]
    field = row["field"]
    old_value = row["old_value"]
    new_value = row["new_value"]


    if sample_id not in corrected_map:
        raise ValueError(
            f"Correction sample survivor'da yok: "
            f"{sample_id}"
        )


    if field != "scenario":
        raise ValueError(
            f"Beklenmeyen field correction: "
            f"{sample_id} -> {field}"
        )


    if row["decision"] != "METADATA_FIX":
        raise ValueError(
            f"Beklenmeyen correction decision: "
            f"{sample_id} -> {row['decision']}"
        )


    record = corrected_map[
        sample_id
    ]


    actual_old_value = record[
        field
    ]


    if actual_old_value != old_value:
        raise ValueError(
            f"Old-value mismatch: {sample_id}\n"
            f"Dataset : {actual_old_value}\n"
            f"Audit   : {old_value}"
        )


    if old_value == new_value:
        raise ValueError(
            f"Old/new aynı: {sample_id}"
        )


    before_snapshot = deepcopy(
        record
    )


    record[field] = new_value


    after_snapshot = deepcopy(
        record
    )


    changed_fields = [
        key
        for key in before_snapshot
        if (
            before_snapshot[key]
            != after_snapshot[key]
        )
    ]


    if changed_fields != ["scenario"]:
        raise ValueError(
            f"Beklenmeyen alan değişikliği: "
            f"{sample_id} -> {changed_fields}"
        )


    audit_rows.append({
        "sample_id":
            sample_id,

        "field":
            field,

        "old_value":
            old_value,

        "new_value":
            new_value,

        "category":
            record["category"],

        "reason_code":
            row["reason_code"],

        "evidence_review_ranks":
            row.get(
                "evidence_review_ranks",
                "",
            ),

        "review_note":
            row["review_note"],

        "review_stage":
            row["review_stage"],

        "applied":
            True,
    })


# ==================================================
# Validate exactly four changed records
# ==================================================

changed_sample_ids = []


for old_record, new_record in zip(
    records,
    corrected_records,
):

    if (
        old_record["sample_id"]
        != new_record["sample_id"]
    ):
        raise ValueError(
            "Record sırası değişti."
        )


    if old_record != new_record:
        changed_sample_ids.append(
            old_record["sample_id"]
        )


if set(changed_sample_ids) != set(
    correction_ids
):
    raise ValueError(
        "Gerçekte değişen kayıtlar correction "
        "dosyasıyla uyuşmuyor."
    )


if len(changed_sample_ids) != 4:
    raise ValueError(
        f"Beklenen 4 değişen kayıt, "
        f"bulunan {len(changed_sample_ids)}."
    )


# ==================================================
# Write corrected survivor
# ==================================================

with SURVIVOR_OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    for record in corrected_records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ==================================================
# Extend Stage ledger
# ==================================================

previous_rows = []

with PREVIOUS_STAGE_LEDGER.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    if not reader.fieldnames:
        raise ValueError(
            "Previous stage ledger header yok."
        )

    for row in reader:
        previous_rows.append(row)


existing_stage_keys = {
    (
        row["sample_id"],
        row["review_stage"],
        row["stage_decision"],
    )
    for row in previous_rows
}


ledger_rows = []


for row in previous_rows:

    ledger_rows.append({
        "sample_id":
            row["sample_id"],

        "review_stage":
            row["review_stage"],

        "review_type":
            row["review_type"],

        "stage_decision":
            row["stage_decision"],

        "reason_code":
            row["reason_code"],

        "kept_reference":
            row["kept_reference"],

        "review_scope":
            row["review_scope"],

        "group_id":
            row["group_id"],

        "field_name":
            "",

        "old_value":
            "",

        "new_value":
            "",

        "review_note":
            row["review_note"],

        "finalized":
            normalize_bool(
                row["finalized"]
            ),
    })


for row in corrections:

    key = (
        row["sample_id"],
        row["review_stage"],
        row["decision"],
    )


    if key in existing_stage_keys:
        raise ValueError(
            "Metadata correction zaten ledger'da "
            f"mevcut: {key}"
        )


    ledger_rows.append({
        "sample_id":
            row["sample_id"],

        "review_stage":
            row["review_stage"],

        "review_type":
            "SCENARIO_METADATA",

        "stage_decision":
            "METADATA_FIX",

        "reason_code":
            row["reason_code"],

        "kept_reference":
            "",

        "review_scope":
            "SCENARIO_BOUNDARY",

        "group_id":
            "",

        "field_name":
            row["field"],

        "old_value":
            row["old_value"],

        "new_value":
            row["new_value"],

        "review_note":
            row["review_note"],

        "finalized":
            False,
    })


LEDGER_FIELDS = [
    "sample_id",
    "review_stage",
    "review_type",
    "stage_decision",
    "reason_code",
    "kept_reference",
    "review_scope",
    "group_id",
    "field_name",
    "old_value",
    "new_value",
    "review_note",
    "finalized",
]


with STAGE_LEDGER_OUTPUT.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=LEDGER_FIELDS,
    )

    writer.writeheader()

    writer.writerows(
        ledger_rows
    )


# ==================================================
# Applied audit
# ==================================================

AUDIT_FIELDS = [
    "sample_id",
    "field",
    "old_value",
    "new_value",
    "category",
    "reason_code",
    "evidence_review_ranks",
    "review_note",
    "review_stage",
    "applied",
]


with APPLIED_AUDIT_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=AUDIT_FIELDS,
    )

    writer.writeheader()

    writer.writerows(
        audit_rows
    )


# ==================================================
# Distribution checks
# ==================================================

category_before = Counter(
    r["category"]
    for r in records
)

category_after = Counter(
    r["category"]
    for r in corrected_records
)


if category_before != category_after:
    raise ValueError(
        "Scenario metadata correction kategori "
        "dağılımını değiştirdi."
    )


scenario_before = Counter(
    (
        r["category"],
        r["scenario"],
    )
    for r in records
)

scenario_after = Counter(
    (
        r["category"],
        r["scenario"],
    )
    for r in corrected_records
)


# ==================================================
# Summary
# ==================================================

summary = {
    "status":
        "SCENARIO_METADATA_FIX_APPLIED_TO_QA_VIEW",

    "input_survivor_count":
        len(records),

    "output_survivor_count":
        len(corrected_records),

    "metadata_correction_count":
        len(corrections),

    "changed_record_count":
        len(changed_sample_ids),

    "changed_sample_ids":
        sorted(
            changed_sample_ids
        ),

    "category_distribution_unchanged":
        category_before
        == category_after,

    "category_distribution":
        dict(
            category_after
        ),

    "input_survivor_sha256":
        sha256_file(
            SURVIVOR_INPUT
        ),

    "corrections_sha256":
        sha256_file(
            CORRECTIONS_FILE
        ),

    "output_survivor_sha256":
        sha256_file(
            SURVIVOR_OUTPUT
        ),

    "master_dataset_modified":
        False,

    "record_count_changed":
        False,

    "policy": (
        "Only audited scenario metadata corrections "
        "were applied to the current QA survivor view. "
        "complaint_text and category were not modified. "
        "The source master dataset remains immutable. "
        "Scenario is QA metadata and is not the model "
        "target label."
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
print("STAGE 7D-2B - APPLY SCENARIO METADATA FIXES")
print("=" * 78)

print(
    f"Input survivor          : "
    f"{len(records)}"
)

print(
    f"Metadata correction     : "
    f"{len(corrections)}"
)

print(
    f"Changed records         : "
    f"{len(changed_sample_ids)}"
)

print(
    f"Output survivor         : "
    f"{len(corrected_records)}"
)

print(
    f"Category unchanged      : "
    f"{category_before == category_after}"
)


print("\nApplied corrections:")

for row in audit_rows:

    print(
        f"  {row['sample_id']}"
    )

    print(
        f"    {row['old_value']} "
        f"-> {row['new_value']}"
    )


print("\nKategori dağılımı:")

for category in sorted(
    category_after
):

    print(
        f"  {category:<24}: "
        f"{category_after[category]}"
    )


print("\nDosyalar:")

print(
    f"  Survivor : "
    f"{SURVIVOR_OUTPUT}"
)

print(
    f"  Ledger   : "
    f"{STAGE_LEDGER_OUTPUT}"
)

print(
    f"  Audit    : "
    f"{APPLIED_AUDIT_FILE}"
)

print(
    f"  Summary  : "
    f"{SUMMARY_FILE}"
)