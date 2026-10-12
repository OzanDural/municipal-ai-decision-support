import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


MASTER_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

CURRENT_SURVIVOR_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d2b_survivor_after_scenario_fix.jsonl"
)

PREVIOUS_LEDGER_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d2b_stage_review_ledger.csv"
)

CROSS_DECISIONS_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d3_cross_category_record_decisions.csv"
)

PAIR_AUDIT_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d3_cross_category_pair_audit.csv"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

LEDGER_OUTPUT = REPORT_DIR / (
    "07d3b_stage_review_ledger.csv"
)

SURVIVOR_OUTPUT = REPORT_DIR / (
    "07d3b_survivor_after_cross_category.jsonl"
)

STATUS_OUTPUT = REPORT_DIR / (
    "07d3b_current_review_status.csv"
)

APPLY_AUDIT_OUTPUT = REPORT_DIR / (
    "07d3b_cross_category_apply_audit.csv"
)

REJECT_IDS_OUTPUT = REPORT_DIR / (
    "07d3b_effective_reject_ids.json"
)

SUMMARY_OUTPUT = REPORT_DIR / (
    "07d3b_review_state_summary.json"
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
# Master
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
    r["sample_id"]: r
    for r in master_records
}


if len(master_map) != 1750:

    raise ValueError(
        "Master içinde duplicate sample_id var."
    )


# ==================================================
# Current survivor
#
# Bu dosyada 4 scenario metadata fix'i
# zaten uygulanmış durumda.
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


if len(survivor_records) != 1681:

    raise ValueError(
        f"Beklenen survivor 1681, "
        f"bulunan {len(survivor_records)}."
    )


survivor_map = {
    r["sample_id"]: r
    for r in survivor_records
}


if len(survivor_map) != 1681:

    raise ValueError(
        "Current survivor içinde "
        "duplicate sample_id var."
    )


survivor_ids = set(
    survivor_map
)


# ==================================================
# Previous Stage 7 ledger
# ==================================================

previous_rows = []

with PREVIOUS_LEDGER_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    if not reader.fieldnames:

        raise ValueError(
            "Previous ledger header bulunamadı."
        )

    for row in reader:

        previous_rows.append(
            row
        )


previous_reject_ids = {
    row["sample_id"]
    for row in previous_rows
    if row[
        "stage_decision"
    ].startswith(
        "REJECT_"
    )
}


if len(previous_reject_ids) != 69:

    raise ValueError(
        f"Beklenen önceki reject 69, "
        f"bulunan "
        f"{len(previous_reject_ids)}."
    )


# Current survivor ile önceki reject
# birbirinden tamamen ayrılmış olmalı.
bad_previous_survivors = (
    previous_reject_ids
    & survivor_ids
)


if bad_previous_survivors:

    raise ValueError(
        "Önceki reject kayıtlarından bazıları "
        "current survivor içinde bulundu:\n"
        + json.dumps(
            sorted(
                bad_previous_survivors
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


# ==================================================
# Cross-category decisions
# ==================================================

new_decisions = []

with CROSS_DECISIONS_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    required_fields = {
        "sample_id",
        "category",
        "scenario",
        "decision",
        "reason_code",
        "review_note",
        "review_stage",
        "finalized",
    }


    if not reader.fieldnames:

        raise ValueError(
            "Cross decision header bulunamadı."
        )


    missing = (
        required_fields
        - set(reader.fieldnames)
    )


    if missing:

        raise ValueError(
            f"Eksik cross-decision kolonları: "
            f"{sorted(missing)}"
        )


    for row in reader:

        new_decisions.append(
            row
        )


if len(new_decisions) != 14:

    raise ValueError(
        f"Beklenen 14 yeni decision, "
        f"bulunan {len(new_decisions)}."
    )


new_decision_ids = [
    row["sample_id"]
    for row in new_decisions
]


if len(
    set(new_decision_ids)
) != 14:

    raise ValueError(
        "Cross decision dosyasında "
        "duplicate sample_id var."
    )


allowed_decisions = {
    "REJECT_ARTIFICIAL",
    "REJECT_AMBIGUOUS",
}


for row in new_decisions:

    sample_id = row[
        "sample_id"
    ]


    if sample_id not in master_map:

        raise ValueError(
            f"Master'da olmayan sample_id: "
            f"{sample_id}"
        )


    if sample_id not in survivor_ids:

        raise ValueError(
            f"Cross-category reject current "
            f"survivor'da yok: {sample_id}"
        )


    if row[
        "decision"
    ] not in allowed_decisions:

        raise ValueError(
            f"Beklenmeyen decision: "
            f"{sample_id} -> "
            f"{row['decision']}"
        )


    current_record = survivor_map[
        sample_id
    ]


    if (
        current_record["category"]
        != row["category"]
    ):

        raise ValueError(
            f"Category mismatch: "
            f"{sample_id}"
        )


    if (
        current_record["scenario"]
        != row["scenario"]
    ):

        raise ValueError(
            f"Scenario mismatch: "
            f"{sample_id}\n"
            f"Survivor : "
            f"{current_record['scenario']}\n"
            f"Decision : "
            f"{row['scenario']}"
        )


# Yeni decision'lar önceki rejectlerle
# overlap etmemeli; zaten survivor üzerinde
# hazırlanmışlardı.
new_reject_ids = set(
    new_decision_ids
)


overlap = (
    previous_reject_ids
    & new_reject_ids
)


if overlap:

    raise ValueError(
        "Yeni cross-category reject ile "
        "önceki reject overlap bulundu:\n"
        + json.dumps(
            sorted(overlap),
            ensure_ascii=False,
            indent=2,
        )
    )


decision_counts = Counter(
    row["decision"]
    for row in new_decisions
)


if (
    decision_counts[
        "REJECT_ARTIFICIAL"
    ] != 11
):

    raise ValueError(
        "Beklenen REJECT_ARTIFICIAL=11."
    )


if (
    decision_counts[
        "REJECT_AMBIGUOUS"
    ] != 3
):

    raise ValueError(
        "Beklenen REJECT_AMBIGUOUS=3."
    )


# ==================================================
# Pair audit integrity
# ==================================================

pair_rows = []

with PAIR_AUDIT_FILE.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        pair_rows.append(
            row
        )


if len(pair_rows) != 92:

    raise ValueError(
        f"Beklenen 92 cross-category pair, "
        f"bulunan {len(pair_rows)}."
    )


keep_both_pair_count = sum(
    1
    for row in pair_rows
    if row[
        "human_decision"
    ]
    == "KEEP_BOTH_VALID_BOUNDARY"
)


affected_pair_count = (
    len(pair_rows)
    - keep_both_pair_count
)


if keep_both_pair_count != 66:

    raise ValueError(
        f"Beklenen KEEP_BOTH pair=66, "
        f"bulunan "
        f"{keep_both_pair_count}."
    )


if affected_pair_count != 26:

    raise ValueError(
        f"Beklenen reject'ten etkilenen "
        f"pair=26, bulunan "
        f"{affected_pair_count}."
    )


# ==================================================
# Extend Stage 7 ledger
# ==================================================

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
            row["field_name"],

        "old_value":
            row["old_value"],

        "new_value":
            row["new_value"],

        "review_note":
            row["review_note"],

        "finalized":
            normalize_bool(
                row["finalized"]
            ),
    })


existing_keys = {
    (
        row["sample_id"],
        row["review_stage"],
        row["stage_decision"],
    )
    for row in ledger_rows
}


for row in new_decisions:

    key = (
        row["sample_id"],
        row["review_stage"],
        row["decision"],
    )


    if key in existing_keys:

        raise ValueError(
            "Yeni decision zaten Stage "
            f"ledger'da mevcut: {key}"
        )


    ledger_rows.append({
        "sample_id":
            row["sample_id"],

        "review_stage":
            row["review_stage"],

        "review_type":
            "CROSS_CATEGORY",

        "stage_decision":
            row["decision"],

        "reason_code":
            row["reason_code"],

        "kept_reference":
            "",

        "review_scope":
            "CROSS_CATEGORY_BOUNDARY",

        "group_id":
            "",

        "field_name":
            "",

        "old_value":
            "",

        "new_value":
            "",

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


with LEDGER_OUTPUT.open(
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
# Effective reject state
# ==================================================

effective_reject_ids = {
    row["sample_id"]
    for row in ledger_rows
    if row[
        "stage_decision"
    ].startswith(
        "REJECT_"
    )
}


if len(effective_reject_ids) != 83:

    raise ValueError(
        f"Beklenen toplam effective reject "
        f"69 + 14 = 83, bulunan "
        f"{len(effective_reject_ids)}."
    )


# ==================================================
# New survivor
#
# Scenario fixes korunur; burada yalnızca
# 14 yeni reject current survivor'dan çıkar.
# ==================================================

output_survivors = [
    record
    for record in survivor_records
    if record[
        "sample_id"
    ]
    not in new_reject_ids
]


if len(output_survivors) != 1667:

    raise ValueError(
        f"Beklenen output survivor 1667, "
        f"bulunan {len(output_survivors)}."
    )


output_ids = {
    r["sample_id"]
    for r in output_survivors
}


if (
    output_ids
    & effective_reject_ids
):

    raise ValueError(
        "Output survivor içinde reject "
        "edilmiş sample bulundu."
    )


with SURVIVOR_OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    for record in output_survivors:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ==================================================
# Verify previous scenario metadata fixes survived
# ==================================================

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


output_map = {
    r["sample_id"]: r
    for r in output_survivors
}


for sample_id, expected_scenario in (
    expected_scenario_fixes.items()
):

    if sample_id not in output_map:

        raise ValueError(
            f"Scenario-fix kaydı "
            f"beklenmedik şekilde survivor'dan "
            f"çıktı: {sample_id}"
        )


    actual_scenario = output_map[
        sample_id
    ][
        "scenario"
    ]


    if actual_scenario != expected_scenario:

        raise ValueError(
            f"Scenario fix korunmadı: "
            f"{sample_id}\n"
            f"Beklenen: {expected_scenario}\n"
            f"Bulunan : {actual_scenario}"
        )


# ==================================================
# Apply audit
# ==================================================

apply_audit_rows = []


for row in new_decisions:

    sample_id = row[
        "sample_id"
    ]


    apply_audit_rows.append({
        "sample_id":
            sample_id,

        "category":
            row["category"],

        "scenario":
            row["scenario"],

        "decision":
            row["decision"],

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

        "removed_from_survivor":
            sample_id
            not in output_ids,
    })


AUDIT_FIELDS = [
    "sample_id",
    "category",
    "scenario",
    "decision",
    "reason_code",
    "evidence_review_ranks",
    "review_note",
    "review_stage",
    "removed_from_survivor",
]


with APPLY_AUDIT_OUTPUT.open(
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
        apply_audit_rows
    )


# ==================================================
# Current status for all master records
# ==================================================

reviews_by_sample = defaultdict(
    list
)


for row in ledger_rows:

    reviews_by_sample[
        row["sample_id"]
    ].append(
        row
    )


status_rows = []


for master_record in master_records:

    sample_id = master_record[
        "sample_id"
    ]

    reviews = reviews_by_sample.get(
        sample_id,
        [],
    )


    rejects = [
        row
        for row in reviews
        if row[
            "stage_decision"
        ].startswith(
            "REJECT_"
        )
    ]


    metadata_fixes = [
        row
        for row in reviews
        if row[
            "stage_decision"
        ]
        == "METADATA_FIX"
    ]


    if rejects:

        effective_status = (
            "REJECT_PROVISIONAL"
        )

    elif sample_id in output_ids:

        effective_status = (
            "SURVIVE_STAGE7_SO_FAR"
        )

    else:

        raise ValueError(
            f"Beklenmeyen status: "
            f"{sample_id}"
        )


    status_rows.append({
        "sample_id":
            sample_id,

        "original_category":
            master_record["category"],

        "original_scenario":
            master_record["scenario"],

        "current_scenario":
            (
                output_map[
                    sample_id
                ][
                    "scenario"
                ]
                if sample_id
                in output_map
                else master_record[
                    "scenario"
                ]
            ),

        "effective_status":
            effective_status,

        "reject_decisions":
            "|".join(
                sorted({
                    row[
                        "stage_decision"
                    ]
                    for row in rejects
                })
            ),

        "reject_reason_codes":
            "|".join(
                sorted({
                    row[
                        "reason_code"
                    ]
                    for row in rejects
                })
            ),

        "metadata_fix_count":
            len(
                metadata_fixes
            ),

        "review_stage_count":
            len(
                {
                    row[
                        "review_stage"
                    ]
                    for row in reviews
                }
            ),
    })


STATUS_FIELDS = [
    "sample_id",
    "original_category",
    "original_scenario",
    "current_scenario",
    "effective_status",
    "reject_decisions",
    "reject_reason_codes",
    "metadata_fix_count",
    "review_stage_count",
]


with STATUS_OUTPUT.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=STATUS_FIELDS,
    )

    writer.writeheader()

    writer.writerows(
        status_rows
    )


# ==================================================
# Effective reject snapshot
# ==================================================

reject_snapshot = []


for sample_id in sorted(
    effective_reject_ids
):

    rejection_reviews = [
        row
        for row in reviews_by_sample[
            sample_id
        ]
        if row[
            "stage_decision"
        ].startswith(
            "REJECT_"
        )
    ]


    reject_snapshot.append({
        "sample_id":
            sample_id,

        "category":
            master_map[
                sample_id
            ][
                "category"
            ],

        "decisions":
            sorted({
                row[
                    "stage_decision"
                ]
                for row
                in rejection_reviews
            }),

        "reason_codes":
            sorted({
                row[
                    "reason_code"
                ]
                for row
                in rejection_reviews
            }),
    })


with REJECT_IDS_OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        reject_snapshot,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ==================================================
# Distribution
# ==================================================

category_before = Counter(
    r["category"]
    for r in survivor_records
)

category_after = Counter(
    r["category"]
    for r in output_survivors
)


category_rejects = Counter(
    survivor_map[
        sample_id
    ][
        "category"
    ]
    for sample_id
    in new_reject_ids
)


# ==================================================
# Global rejection counts by decision
# ==================================================

global_reject_decisions = Counter(
    row["stage_decision"]
    for row in ledger_rows
    if row[
        "stage_decision"
    ].startswith(
        "REJECT_"
    )
)


# There must be:
# 58 duplicate
# 22 artificial total (11 earlier + 11 new)
# 3 ambiguous
if global_reject_decisions[
    "REJECT_DUPLICATE"
] != 58:

    raise ValueError(
        "Global REJECT_DUPLICATE != 58"
    )


if global_reject_decisions[
    "REJECT_ARTIFICIAL"
] != 22:

    raise ValueError(
        "Global REJECT_ARTIFICIAL != 22"
    )


if global_reject_decisions[
    "REJECT_AMBIGUOUS"
] != 3:

    raise ValueError(
        "Global REJECT_AMBIGUOUS != 3"
    )


# ==================================================
# Summary
# ==================================================

summary = {
    "status":
        "STAGE7_REVIEW_STATE_AFTER_CROSS_CATEGORY",

    "master_record_count":
        len(master_records),

    "input_survivor_count":
        len(survivor_records),

    "new_cross_category_reject_count":
        len(new_reject_ids),

    "new_reject_artificial_count":
        decision_counts[
            "REJECT_ARTIFICIAL"
        ],

    "new_reject_ambiguous_count":
        decision_counts[
            "REJECT_AMBIGUOUS"
        ],

    "previous_effective_reject_count":
        len(previous_reject_ids),

    "effective_reject_count":
        len(effective_reject_ids),

    "output_survivor_count":
        len(output_survivors),

    "cross_category_pairs_reviewed":
        len(pair_rows),

    "valid_hard_negative_pairs_kept":
        keep_both_pair_count,

    "pairs_affected_by_reject":
        affected_pair_count,

    "global_reject_decision_counts":
        dict(
            global_reject_decisions
        ),

    "category_reject_counts_this_stage":
        dict(
            category_rejects
        ),

    "category_distribution_before":
        dict(
            category_before
        ),

    "category_distribution_after":
        dict(
            category_after
        ),

    "scenario_fixes_preserved":
        True,

    "target_label_correction_count":
        0,

    "master_dataset_modified":
        False,

    "survivor_is_final_dataset":
        False,

    "master_sha256":
        sha256_file(
            MASTER_FILE
        ),

    "input_survivor_sha256":
        sha256_file(
            CURRENT_SURVIVOR_FILE
        ),

    "cross_decisions_sha256":
        sha256_file(
            CROSS_DECISIONS_FILE
        ),

    "output_survivor_sha256":
        sha256_file(
            SURVIVOR_OUTPUT
        ),

    "policy": (
        "Cross-category review decisions were "
        "appended to the stage-specific QA ledger. "
        "Valid hard-negative boundaries were retained. "
        "Only manually identified artificial or "
        "category-ambiguous records were removed from "
        "the current survivor view. Previous audited "
        "scenario metadata fixes were preserved. "
        "The immutable master dataset was not modified."
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


# ==================================================
# Terminal
# ==================================================

print("=" * 78)
print("STAGE 7D-3B - MERGE CROSS-CATEGORY DECISIONS")
print("=" * 78)

print(
    f"Master kayıt              : "
    f"{len(master_records)}"
)

print(
    f"Input survivor            : "
    f"{len(survivor_records)}"
)

print(
    f"Previous reject           : "
    f"{len(previous_reject_ids)}"
)


print("\nYeni cross-category kararları:")

print(
    f"  REJECT_ARTIFICIAL       : "
    f"{decision_counts['REJECT_ARTIFICIAL']}"
)

print(
    f"  REJECT_AMBIGUOUS        : "
    f"{decision_counts['REJECT_AMBIGUOUS']}"
)

print(
    f"  Toplam yeni reject      : "
    f"{len(new_reject_ids)}"
)


print("\nGlobal Stage 7 reject:")

for key, value in sorted(
    global_reject_decisions.items()
):

    print(
        f"  {key:<24}: {value}"
    )


print(
    f"\nEffective reject          : "
    f"{len(effective_reject_ids)}"
)

print(
    f"Output survivor           : "
    f"{len(output_survivors)}"
)


print("\nCross-category audit:")

print(
    f"  Reviewed pair           : "
    f"{len(pair_rows)}"
)

print(
    f"  Valid KEEP_BOTH         : "
    f"{keep_both_pair_count}"
)

print(
    f"  Reject affected pair    : "
    f"{affected_pair_count}"
)

print(
    "  Target label correction: 0"
)


print("\nScenario fixes preserved  : True")


print("\nKategori dağılımı:")

for category in sorted(
    category_after
):

    print(
        f"  {category:<24} "
        f"{category_before[category]} "
        f"-> "
        f"{category_after[category]}"
    )


print("\nDosyalar:")

print(
    f"  Ledger    : {LEDGER_OUTPUT}"
)

print(
    f"  Survivor  : {SURVIVOR_OUTPUT}"
)

print(
    f"  Status    : {STATUS_OUTPUT}"
)

print(
    f"  ApplyAudit: {APPLY_AUDIT_OUTPUT}"
)

print(
    f"  RejectIDs : {REJECT_IDS_OUTPUT}"
)

print(
    f"  Summary   : {SUMMARY_OUTPUT}"
)