import csv
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


MASTER_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

CURRENT_SURVIVOR_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d3b_survivor_after_cross_category.jsonl"
)

STAGE_LEDGER_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d3b_stage_review_ledger.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

FINAL_ACCEPTED_FILE = OUTPUT_DIR / (
    "master_final.jsonl"
)

FINAL_REJECTED_FILE = OUTPUT_DIR / (
    "master_rejected.jsonl"
)

FINAL_LEDGER_FILE = REPORT_DIR / (
    "08_final_review_ledger.csv"
)

FINAL_AUDIT_FILE = REPORT_DIR / (
    "08_final_decision_audit.csv"
)

SUMMARY_FILE = REPORT_DIR / (
    "08_final_dataset_summary.json"
)


EXPECTED_MASTER_SHA256 = (
    "eadcf4af5ee4dd4afb6742acbd70919541d8240c0e9dbe40f30e99fb8f9f0e29"
)


# ==================================================
# Helpers
# ==================================================

def sha256_file(path):

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


def normalize_bool(value):

    if isinstance(value, bool):
        return value

    value = str(
        value
    ).strip().lower()

    if value in {
        "true",
        "1",
        "yes",
    }:
        return True

    if value in {
        "false",
        "0",
        "no",
        "",
    }:
        return False

    raise ValueError(
        f"Geçersiz boolean: {value}"
    )


# ==================================================
# Freeze/check Stage 2 master
# ==================================================

master_hash = sha256_file(
    MASTER_FILE
)


if master_hash != EXPECTED_MASTER_SHA256:

    raise ValueError(
        "master_stage2.jsonl hash değişmiş!\n"
        f"Beklenen : {EXPECTED_MASTER_SHA256}\n"
        f"Bulunan  : {master_hash}\n"
        "Final dataset oluşturulmadı."
    )


# ==================================================
# Load immutable master
# ==================================================

master_records = []

with MASTER_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        master_records.append(
            json.loads(line)
        )


if len(master_records) != 1750:

    raise ValueError(
        f"Beklenen master 1750, "
        f"bulunan {len(master_records)}."
    )


master_map = {
    record["sample_id"]:
        record
    for record in master_records
}


if len(master_map) != 1750:

    raise ValueError(
        "Master dataset içinde duplicate "
        "sample_id bulundu."
    )


# Master QA state'in hâlâ PENDING
# olması bekleniyor.
qa_status_before = Counter(
    record["qa_status"]
    for record in master_records
)


if qa_status_before != {
    "PENDING": 1750
}:

    raise ValueError(
        "master_stage2 qa_status beklenmeyen "
        f"durumda: {dict(qa_status_before)}"
    )


# ==================================================
# Load Stage 7 survivor
# ==================================================

survivor_records = []

with CURRENT_SURVIVOR_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        survivor_records.append(
            json.loads(line)
        )


if len(survivor_records) != 1667:

    raise ValueError(
        f"Beklenen Stage7 survivor 1667, "
        f"bulunan {len(survivor_records)}."
    )


survivor_map = {
    record["sample_id"]:
        record
    for record in survivor_records
}


if len(survivor_map) != 1667:

    raise ValueError(
        "Stage7 survivor içinde duplicate "
        "sample_id bulundu."
    )


# ==================================================
# Load complete Stage 7 ledger
# ==================================================

stage_rows = []

with STAGE_LEDGER_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    if not reader.fieldnames:

        raise ValueError(
            "Stage 7 ledger header bulunamadı."
        )

    for row in reader:

        stage_rows.append(
            row
        )


# Beklenen stage-level karar toplamı:
#
# KEEP                  104
# REJECT_DUPLICATE       58
# REJECT_ARTIFICIAL      22
# REJECT_AMBIGUOUS        3
# METADATA_FIX            4
# --------------------------
# TOTAL                  191

if len(stage_rows) != 191:

    raise ValueError(
        f"Beklenen ledger row 191, "
        f"bulunan {len(stage_rows)}."
    )


stage_decision_counts = Counter(
    row["stage_decision"]
    for row in stage_rows
)


expected_stage_counts = {
    "KEEP": 104,
    "REJECT_DUPLICATE": 58,
    "REJECT_ARTIFICIAL": 22,
    "REJECT_AMBIGUOUS": 3,
    "METADATA_FIX": 4,
}


if dict(
    stage_decision_counts
) != expected_stage_counts:

    raise ValueError(
        "Stage decision dağılımı beklenenden "
        "farklı:\n"
        + json.dumps(
            dict(stage_decision_counts),
            ensure_ascii=False,
            indent=2,
        )
    )


# ==================================================
# Reviews by sample
# ==================================================

reviews_by_sample = defaultdict(
    list
)


for row in stage_rows:

    sample_id = row[
        "sample_id"
    ]

    if sample_id not in master_map:

        raise ValueError(
            f"Ledger sample master'da yok: "
            f"{sample_id}"
        )

    reviews_by_sample[
        sample_id
    ].append(
        row
    )


# ==================================================
# Effective rejects
# ==================================================

reject_rows_by_sample = defaultdict(
    list
)


for row in stage_rows:

    if row[
        "stage_decision"
    ].startswith(
        "REJECT_"
    ):

        reject_rows_by_sample[
            row["sample_id"]
        ].append(
            row
        )


reject_ids = set(
    reject_rows_by_sample.keys()
)


if len(reject_ids) != 83:

    raise ValueError(
        f"Beklenen unique reject 83, "
        f"bulunan {len(reject_ids)}."
    )


# Şu anki pipeline'da bir kayıt
# birden fazla reject reason taşımıyor.
multiple_reject_decisions = {
    sample_id:
        rows
    for sample_id, rows
    in reject_rows_by_sample.items()
    if len(rows) > 1
}


if multiple_reject_decisions:

    raise ValueError(
        "Bir kayıtta birden fazla reject "
        "decision bulundu. Manuel kontrol gerekli:\n"
        + json.dumps(
            {
                key: [
                    row["stage_decision"]
                    for row in value
                ]
                for key, value
                in multiple_reject_decisions.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# ==================================================
# Reject type integrity
# ==================================================

unique_reject_decisions = Counter(
    rows[0]["stage_decision"]
    for rows
    in reject_rows_by_sample.values()
)


expected_reject_counts = {
    "REJECT_DUPLICATE": 58,
    "REJECT_ARTIFICIAL": 22,
    "REJECT_AMBIGUOUS": 3,
}


if dict(
    unique_reject_decisions
) != expected_reject_counts:

    raise ValueError(
        "Unique reject decision count "
        "beklenenden farklı:\n"
        + json.dumps(
            dict(
                unique_reject_decisions
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


# ==================================================
# Scenario metadata fixes
# ==================================================

metadata_fix_rows = [
    row
    for row in stage_rows
    if row[
        "stage_decision"
    ] == "METADATA_FIX"
]


if len(metadata_fix_rows) != 4:

    raise ValueError(
        f"Beklenen metadata fix 4, "
        f"bulunan {len(metadata_fix_rows)}."
    )


scenario_fix_map = {}


for row in metadata_fix_rows:

    sample_id = row[
        "sample_id"
    ]

    field_name = row[
        "field_name"
    ]


    if field_name != "scenario":

        raise ValueError(
            f"Beklenmeyen metadata field: "
            f"{sample_id} -> {field_name}"
        )


    if sample_id in reject_ids:

        raise ValueError(
            "Metadata-fix kaydı aynı zamanda "
            f"reject edilmiş: {sample_id}"
        )


    if sample_id in scenario_fix_map:

        raise ValueError(
            "Aynı sample için birden fazla "
            f"scenario fix var: {sample_id}"
        )


    old_value = row[
        "old_value"
    ]

    new_value = row[
        "new_value"
    ]


    if (
        master_map[
            sample_id
        ][
            "scenario"
        ]
        != old_value
    ):

        raise ValueError(
            f"Scenario old-value mismatch: "
            f"{sample_id}"
        )


    scenario_fix_map[
        sample_id
    ] = {
        "old_value":
            old_value,

        "new_value":
            new_value,

        "reason_code":
            row["reason_code"],

        "review_note":
            row["review_note"],
    }


expected_scenario_fixes = {
    "GPT_long_gpt_001_069":
        "unauthorized_signage_display",

    "GPT_long_gpt_003_005":
        "overflowing_container",

    "CLAUDE_prod_claude_002_013":
        "road_surface_collapse",

    "GPT_prod_gpt_002_071":
        "business_sidewalk_occupation",
}


actual_scenario_fixes = {
    sample_id:
        data["new_value"]
    for sample_id, data
    in scenario_fix_map.items()
}


if (
    actual_scenario_fixes
    != expected_scenario_fixes
):

    raise ValueError(
        "Scenario fix set beklenenden farklı:\n"
        + json.dumps(
            actual_scenario_fixes,
            ensure_ascii=False,
            indent=2,
        )
    )


# ==================================================
# Reconstruct curated dataset from immutable master
# ==================================================

curated_records = [
    deepcopy(record)
    for record in master_records
]


curated_map = {
    record["sample_id"]:
        record
    for record in curated_records
}


# Apply only approved metadata fixes.
for sample_id, fix in (
    scenario_fix_map.items()
):

    curated_map[
        sample_id
    ][
        "scenario"
    ] = fix[
        "new_value"
    ]


# ==================================================
# Build pre-final accepted/rejected views
# ==================================================

accepted_pre_final = [
    record
    for record in curated_records
    if record[
        "sample_id"
    ]
    not in reject_ids
]


rejected_pre_final = [
    record
    for record in curated_records
    if record[
        "sample_id"
    ]
    in reject_ids
]


if len(
    accepted_pre_final
) != 1667:

    raise ValueError(
        f"Beklenen accepted 1667, "
        f"bulunan "
        f"{len(accepted_pre_final)}."
    )


if len(
    rejected_pre_final
) != 83:

    raise ValueError(
        f"Beklenen rejected 83, "
        f"bulunan "
        f"{len(rejected_pre_final)}."
    )


# ==================================================
# Strong consistency check:
#
# Reconstructed accepted view must exactly equal
# Stage 7 survivor BEFORE qa_status finalization.
# ==================================================

if (
    len(accepted_pre_final)
    != len(survivor_records)
):

    raise ValueError(
        "Reconstructed accepted count "
        "Stage7 survivor ile uyuşmuyor."
    )


for reconstructed, stage7_record in zip(
    accepted_pre_final,
    survivor_records,
):

    if (
        reconstructed["sample_id"]
        != stage7_record["sample_id"]
    ):

        raise ValueError(
            "Stage7 survivor record sırası "
            "reconstruction ile uyuşmuyor."
        )


    if reconstructed != stage7_record:

        raise ValueError(
            "Stage7 survivor içerik mismatch:\n"
            f"{reconstructed['sample_id']}"
        )


# ==================================================
# Final QA statuses
# ==================================================

final_accepted = [
    deepcopy(record)
    for record in accepted_pre_final
]


final_rejected = [
    deepcopy(record)
    for record in rejected_pre_final
]


for record in final_accepted:

    record[
        "qa_status"
    ] = "ACCEPTED"


for record in final_rejected:

    record[
        "qa_status"
    ] = "REJECTED"


# ==================================================
# Verify model target/text never changed
# ==================================================

for record in (
    final_accepted
    + final_rejected
):

    sample_id = record[
        "sample_id"
    ]

    original = master_map[
        sample_id
    ]


    if (
        record[
            "complaint_text"
        ]
        != original[
            "complaint_text"
        ]
    ):

        raise ValueError(
            "complaint_text değişmiş: "
            f"{sample_id}"
        )


    if (
        record["category"]
        != original["category"]
    ):

        raise ValueError(
            "CATEGORY DEĞİŞMİŞ: "
            f"{sample_id}"
        )


# ==================================================
# Verify only scenario + qa_status can differ
# from immutable master
# ==================================================

allowed_changed_fields = {
    "qa_status",
    "scenario",
}


observed_scenario_changes = set()


for record in (
    final_accepted
    + final_rejected
):

    sample_id = record[
        "sample_id"
    ]

    original = master_map[
        sample_id
    ]


    changed_fields = {
        key
        for key in original
        if (
            original[key]
            != record[key]
        )
    }


    if not changed_fields.issubset(
        allowed_changed_fields
    ):

        raise ValueError(
            f"Beklenmeyen alan değişikliği "
            f"{sample_id}: "
            f"{sorted(changed_fields)}"
        )


    if (
        original["scenario"]
        != record["scenario"]
    ):

        observed_scenario_changes.add(
            sample_id
        )


if (
    observed_scenario_changes
    != set(
        scenario_fix_map.keys()
    )
):

    raise ValueError(
        "Final dataset'teki scenario "
        "değişiklikleri correction ledger ile "
        "uyuşmuyor."
    )


# ==================================================
# Final category counts
# ==================================================

accepted_category_counts = Counter(
    record["category"]
    for record in final_accepted
)


expected_accepted_categories = {
    "HAYVAN_VETERINER": 336,
    "KENT_DUZENI_DENETIM": 332,
    "PARK_YESIL_ALAN": 328,
    "TEMIZLIK_ATIK": 338,
    "YOL_KALDIRIM": 333,
}


if dict(
    accepted_category_counts
) != expected_accepted_categories:

    raise ValueError(
        "Final accepted kategori dağılımı "
        "beklenenden farklı:\n"
        + json.dumps(
            dict(
                accepted_category_counts
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


rejected_category_counts = Counter(
    record["category"]
    for record in final_rejected
)


expected_rejected_categories = {
    "HAYVAN_VETERINER": 14,
    "KENT_DUZENI_DENETIM": 18,
    "PARK_YESIL_ALAN": 22,
    "TEMIZLIK_ATIK": 12,
    "YOL_KALDIRIM": 17,
}


if dict(
    rejected_category_counts
) != expected_rejected_categories:

    raise ValueError(
        "Final rejected kategori dağılımı "
        "beklenenden farklı:\n"
        + json.dumps(
            dict(
                rejected_category_counts
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


# ==================================================
# Conservation checks
# ==================================================

accepted_ids = {
    r["sample_id"]
    for r in final_accepted
}

rejected_ids_final = {
    r["sample_id"]
    for r in final_rejected
}


if (
    accepted_ids
    & rejected_ids_final
):

    raise ValueError(
        "Accepted/rejected overlap var."
    )


if (
    accepted_ids
    | rejected_ids_final
) != set(
    master_map.keys()
):

    raise ValueError(
        "Accepted + rejected master'ın "
        "tamamını kapsamıyor."
    )


if (
    len(final_accepted)
    + len(final_rejected)
    != 1750
):

    raise ValueError(
        "Record conservation başarısız."
    )


# ==================================================
# Generator distributions
# ==================================================

accepted_generator_counts = Counter(
    record["generator_id"]
    for record in final_accepted
)


rejected_generator_counts = Counter(
    record["generator_id"]
    for record in final_rejected
)


# ==================================================
# Source-length family distribution
# ==================================================

def source_family(record):

    batch = record[
        "generation_batch"
    ]

    if batch.startswith(
        "long_"
    ):
        return "LONG"

    return "ORIGINAL"


accepted_source_family = Counter(
    source_family(record)
    for record in final_accepted
)


rejected_source_family = Counter(
    source_family(record)
    for record in final_rejected
)


# ==================================================
# Build final per-record decision audit
# ==================================================

audit_rows = []


for original in master_records:

    sample_id = original[
        "sample_id"
    ]


    if sample_id in reject_ids:

        reject_row = (
            reject_rows_by_sample[
                sample_id
            ][0]
        )

        final_decision = reject_row[
            "stage_decision"
        ]

        reject_reason = reject_row[
            "reason_code"
        ]

        reject_stage = reject_row[
            "review_stage"
        ]

    else:

        final_decision = (
            "ACCEPT"
        )

        reject_reason = ""

        reject_stage = ""


    metadata_fix = (
        scenario_fix_map.get(
            sample_id
        )
    )


    current_scenario = (
        metadata_fix[
            "new_value"
        ]
        if metadata_fix
        else original[
            "scenario"
        ]
    )


    audit_rows.append({
        "sample_id":
            sample_id,

        "category":
            original[
                "category"
            ],

        "original_scenario":
            original[
                "scenario"
            ],

        "final_scenario":
            current_scenario,

        "scenario_metadata_fixed":
            metadata_fix
            is not None,

        "final_decision":
            final_decision,

        "reject_reason_code":
            reject_reason,

        "reject_review_stage":
            reject_stage,

        "metadata_fix_reason_code":
            (
                metadata_fix[
                    "reason_code"
                ]
                if metadata_fix
                else ""
            ),

        "final_qa_status":
            (
                "REJECTED"
                if sample_id
                in reject_ids
                else "ACCEPTED"
            ),
    })


# ==================================================
# Finalize Stage review ledger
# ==================================================

final_ledger_rows = []


for row in stage_rows:

    final_row = dict(
        row
    )

    final_row[
        "finalized"
    ] = True

    final_ledger_rows.append(
        final_row
    )


# ==================================================
# Create output directories
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# Write accepted dataset
# ==================================================

with FINAL_ACCEPTED_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for record in final_accepted:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ==================================================
# Write rejected dataset
# ==================================================

with FINAL_REJECTED_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for record in final_rejected:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ==================================================
# Write final review ledger
# ==================================================

ledger_fields = list(
    stage_rows[0].keys()
)


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

    writer.writerows(
        final_ledger_rows
    )


# ==================================================
# Write final decision audit
# ==================================================

audit_fields = [
    "sample_id",
    "category",
    "original_scenario",
    "final_scenario",
    "scenario_metadata_fixed",
    "final_decision",
    "reject_reason_code",
    "reject_review_stage",
    "metadata_fix_reason_code",
    "final_qa_status",
]


with FINAL_AUDIT_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=audit_fields,
    )

    writer.writeheader()

    writer.writerows(
        audit_rows
    )


# ==================================================
# Freeze hashes
# ==================================================

accepted_hash = sha256_file(
    FINAL_ACCEPTED_FILE
)

rejected_hash = sha256_file(
    FINAL_REJECTED_FILE
)

ledger_hash = sha256_file(
    FINAL_LEDGER_FILE
)

audit_hash = sha256_file(
    FINAL_AUDIT_FILE
)


# ==================================================
# Final summary
# ==================================================

summary = {
    "status":
        "FINAL_DATASET_QA_COMPLETE",

    "master_record_count":
        1750,

    "accepted_record_count":
        1667,

    "rejected_record_count":
        83,

    "record_conservation_pass":
        True,

    "accepted_rejected_overlap":
        0,

    "target_label_correction_count":
        0,

    "complaint_text_correction_count":
        0,

    "scenario_metadata_correction_count":
        4,

    "final_reject_counts":
        dict(
            unique_reject_decisions
        ),

    "accepted_category_distribution":
        dict(
            accepted_category_counts
        ),

    "rejected_category_distribution":
        dict(
            rejected_category_counts
        ),

    "accepted_generator_distribution":
        dict(
            accepted_generator_counts
        ),

    "rejected_generator_distribution":
        dict(
            rejected_generator_counts
        ),

    "accepted_source_family_distribution":
        dict(
            accepted_source_family
        ),

    "rejected_source_family_distribution":
        dict(
            rejected_source_family
        ),

    "qa_status_before":
        dict(
            qa_status_before
        ),

    "qa_status_final_accepted":
        {
            "ACCEPTED":
                len(
                    final_accepted
                )
        },

    "qa_status_final_rejected":
        {
            "REJECTED":
                len(
                    final_rejected
                )
        },

    "immutable_master_sha256":
        master_hash,

    "stage7_survivor_sha256":
        sha256_file(
            CURRENT_SURVIVOR_FILE
        ),

    "final_accepted_sha256":
        accepted_hash,

    "final_rejected_sha256":
        rejected_hash,

    "final_review_ledger_sha256":
        ledger_hash,

    "final_decision_audit_sha256":
        audit_hash,

    "immutable_master_modified":
        False,

    "final_dataset_frozen":
        True,

    "policy": (
        "The final accepted dataset was reconstructed "
        "from the immutable Stage 2 master using the "
        "audited Stage 7 review ledger. Exactly 83 "
        "records were rejected and four approved "
        "scenario metadata corrections were applied. "
        "Complaint text and target category were never "
        "modified. The final accepted dataset is now "
        "frozen for subsequent distribution, tokenizer, "
        "split, leakage, baseline and BERTurk stages."
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
print("STAGE 8 - FINAL DATASET BUILD & GLOBAL AUDIT")
print("=" * 78)

print(
    f"Immutable master       : "
    f"{len(master_records)}"
)

print(
    f"Accepted               : "
    f"{len(final_accepted)}"
)

print(
    f"Rejected               : "
    f"{len(final_rejected)}"
)

print(
    f"Conservation           : "
    f"{len(final_accepted)} + "
    f"{len(final_rejected)} = "
    f"{len(master_records)}"
)


print("\nFinal reject reasons:")

for key, value in sorted(
    unique_reject_decisions.items()
):

    print(
        f"  {key:<24}: {value}"
    )


print("\nCorrections:")

print(
    "  complaint_text       : 0"
)

print(
    "  target category      : 0"
)

print(
    "  scenario metadata    : 4"
)


print("\nAccepted category distribution:")

for category in sorted(
    accepted_category_counts
):

    print(
        f"  {category:<24}: "
        f"{accepted_category_counts[category]}"
    )


print("\nRejected category distribution:")

for category in sorted(
    rejected_category_counts
):

    print(
        f"  {category:<24}: "
        f"{rejected_category_counts[category]}"
    )


print("\nHashes:")

print(
    f"  Stage2 master : "
    f"{master_hash}"
)

print(
    f"  FINAL ACCEPT  : "
    f"{accepted_hash}"
)

print(
    f"  FINAL REJECT  : "
    f"{rejected_hash}"
)


print("\nOutputs:")

print(
    f"  Accepted : "
    f"{FINAL_ACCEPTED_FILE}"
)

print(
    f"  Rejected : "
    f"{FINAL_REJECTED_FILE}"
)

print(
    f"  Ledger   : "
    f"{FINAL_LEDGER_FILE}"
)

print(
    f"  Audit    : "
    f"{FINAL_AUDIT_FILE}"
)

print(
    f"  Summary  : "
    f"{SUMMARY_FILE}"
)


print("\nSTATUS: FINAL_DATASET_QA_COMPLETE")