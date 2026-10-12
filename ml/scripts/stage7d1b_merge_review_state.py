import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


MASTER_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

DUPLICATE_LEDGER_FILE = Path(
    "ml/reports/dataset_qa/"
    "07c_duplicate_review_ledger.csv"
)

ARTIFICIAL_DECISIONS_FILE = Path(
    "ml/reports/dataset_qa/"
    "07d1_artificial_contrast_decisions.csv"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

STAGE_LEDGER_FILE = REPORT_DIR / (
    "07d1b_stage_review_ledger.csv"
)

STATUS_FILE = REPORT_DIR / (
    "07d1b_current_review_status.csv"
)

SURVIVOR_FILE = REPORT_DIR / (
    "07d1b_survivor_after_artificial.jsonl"
)

SUMMARY_FILE = REPORT_DIR / (
    "07d1b_review_state_summary.json"
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

    value = str(value).strip().lower()

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
        f"Geçersiz boolean değer: {value}"
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
        f"Beklenen 1750 master kayıt, "
        f"bulunan {len(records)}."
    )


record_map = {
    r["sample_id"]:
        r
    for r in records
}


record_ids = set(
    record_map.keys()
)


# ==================================================
# Duplicate-stage ledger
# ==================================================

duplicate_rows = []


with DUPLICATE_LEDGER_FILE.open(
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
                f"Duplicate ledger sample_id "
                f"master'da yok: {sample_id}"
            )

        decision = row[
            "decision"
        ]

        if decision not in {
            "KEEP",
            "REJECT_DUPLICATE",
        }:

            raise ValueError(
                f"Geçersiz duplicate kararı: "
                f"{sample_id} -> {decision}"
            )

        duplicate_rows.append(
            row
        )


# Duplicate ledger içinde aynı sample_id için
# çelişkili stage kararı olmamalı.
duplicate_map = {}


for row in duplicate_rows:

    sample_id = row[
        "sample_id"
    ]

    if sample_id in duplicate_map:

        old = duplicate_map[
            sample_id
        ]

        if (
            old["decision"]
            != row["decision"]
        ):

            raise ValueError(
                f"Duplicate ledger çelişkisi: "
                f"{sample_id}"
            )

        continue

    duplicate_map[
        sample_id
    ] = row


duplicate_reject_ids = {
    sample_id
    for sample_id, row
    in duplicate_map.items()
    if row["decision"]
    == "REJECT_DUPLICATE"
}


if len(duplicate_reject_ids) != 58:

    raise ValueError(
        f"Beklenen 58 duplicate reject, "
        f"bulunan "
        f"{len(duplicate_reject_ids)}."
    )


# ==================================================
# Artificial-stage decisions
# ==================================================

artificial_rows = []


with ARTIFICIAL_DECISIONS_FILE.open(
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
                f"Artificial decision sample_id "
                f"master'da yok: {sample_id}"
            )

        if (
            row["decision"]
            != "REJECT_ARTIFICIAL"
        ):

            raise ValueError(
                f"Beklenmeyen artificial karar: "
                f"{sample_id} -> "
                f"{row['decision']}"
            )

        artificial_rows.append(
            row
        )


artificial_map = {}


for row in artificial_rows:

    sample_id = row[
        "sample_id"
    ]

    if sample_id in artificial_map:

        raise ValueError(
            f"Artificial karar dosyasında "
            f"duplicate sample_id: {sample_id}"
        )

    artificial_map[
        sample_id
    ] = row


artificial_reject_ids = set(
    artificial_map.keys()
)


if len(
    artificial_reject_ids
) != 11:

    raise ValueError(
        f"Beklenen 11 artificial reject, "
        f"bulunan "
        f"{len(artificial_reject_ids)}."
    )


# Stage 7D-1 prepare script artificial
# review'ü duplicate survivor üzerinden
# hazırlamıştı. Dolayısıyla burada bir
# artificial kayıt daha önce duplicate
# olarak reject edilmiş olmamalı.
overlap_rejected = (
    duplicate_reject_ids
    & artificial_reject_ids
)


if overlap_rejected:

    raise ValueError(
        "Artificial review kayıtlarından bazıları "
        "zaten REJECT_DUPLICATE durumda:\n"
        + json.dumps(
            sorted(
                overlap_rejected
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


# ==================================================
# Append-only stage ledger
#
# Burada KEEP stage-local'dir.
# Nihai ACCEPT anlamına gelmez.
# ==================================================

stage_rows = []


for sample_id, row in sorted(
    duplicate_map.items()
):

    stage_rows.append({
        "sample_id":
            sample_id,

        "review_stage":
            row[
                "review_stage"
            ],

        "review_type":
            "DUPLICATE",

        "stage_decision":
            row[
                "decision"
            ],

        "reason_code":
            row[
                "reason_code"
            ],

        "kept_reference":
            row[
                "kept_reference"
            ],

        "review_scope":
            row[
                "review_scope"
            ],

        "group_id":
            row[
                "group_id"
            ],

        "review_note":
            row[
                "review_note"
            ],

        "finalized":
            normalize_bool(
                row[
                    "finalized"
                ]
            ),
    })


for sample_id, row in sorted(
    artificial_map.items()
):

    stage_rows.append({
        "sample_id":
            sample_id,

        "review_stage":
            row[
                "review_stage"
            ],

        "review_type":
            "ARTIFICIAL_CONTRAST",

        "stage_decision":
            row[
                "decision"
            ],

        "reason_code":
            row[
                "reason_code"
            ],

        "kept_reference":
            "",

        "review_scope":
            "ARTIFICIAL_CONTRAST",

        "group_id":
            "",

        "review_note":
            row[
                "review_note"
            ],

        "finalized":
            normalize_bool(
                row[
                    "finalized"
                ]
            ),
    })


STAGE_LEDGER_FIELDS = [
    "sample_id",
    "review_stage",
    "review_type",
    "stage_decision",
    "reason_code",
    "kept_reference",
    "review_scope",
    "group_id",
    "review_note",
    "finalized",
]


with STAGE_LEDGER_FILE.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=STAGE_LEDGER_FIELDS,
    )

    writer.writeheader()

    writer.writerows(
        stage_rows
    )


# ==================================================
# Effective current status
#
# REJECT_* herhangi bir stage'de varsa
# kayıt mevcut survivor'dan çıkar.
#
# KEEP ise yalnızca ilgili kontrolden
# geçtiğini gösterir.
# ==================================================

reviews_by_sample = defaultdict(
    list
)


for row in stage_rows:

    reviews_by_sample[
        row["sample_id"]
    ].append(
        row
    )


current_status_rows = []

effective_reject_ids = set()


for record in records:

    sample_id = record[
        "sample_id"
    ]

    reviews = reviews_by_sample.get(
        sample_id,
        [],
    )


    duplicate_decision = ""

    artificial_decision = ""


    if sample_id in duplicate_map:

        duplicate_decision = (
            duplicate_map[
                sample_id
            ][
                "decision"
            ]
        )


    if sample_id in artificial_map:

        artificial_decision = (
            artificial_map[
                sample_id
            ][
                "decision"
            ]
        )


    rejection_rows = [
        row
        for row in reviews
        if row[
            "stage_decision"
        ].startswith(
            "REJECT_"
        )
    ]


    if rejection_rows:

        effective_status = (
            "REJECT_PROVISIONAL"
        )

        effective_reject_ids.add(
            sample_id
        )

    elif reviews:

        effective_status = (
            "SURVIVE_REVIEW_SO_FAR"
        )

    else:

        effective_status = (
            "NOT_YET_STAGE7_REVIEWED"
        )


    reject_reason_codes = sorted(
        {
            row["reason_code"]
            for row
            in rejection_rows
        }
    )


    review_stages = sorted(
        {
            row["review_stage"]
            for row
            in reviews
        }
    )


    current_status_rows.append({
        "sample_id":
            sample_id,

        "category":
            record["category"],

        "scenario":
            record["scenario"],

        "duplicate_decision":
            duplicate_decision,

        "artificial_decision":
            artificial_decision,

        "effective_status":
            effective_status,

        "reject_reason_codes":
            "|".join(
                reject_reason_codes
            ),

        "review_stages":
            "|".join(
                review_stages
            ),
    })


# ==================================================
# Expected current reject state
# ==================================================

if len(
    effective_reject_ids
) != 69:

    raise ValueError(
        f"Beklenen toplam provisional reject "
        f"58 + 11 = 69, bulunan "
        f"{len(effective_reject_ids)}."
    )


# ==================================================
# Survivor view
#
# HÂLÂ FINAL DATASET DEĞİL.
# ==================================================

survivors = [
    record
    for record in records
    if record[
        "sample_id"
    ]
    not in effective_reject_ids
]


if len(survivors) != 1681:

    raise ValueError(
        f"Beklenen survivor 1681, "
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
# Status CSV
# ==================================================

STATUS_FIELDS = [
    "sample_id",
    "category",
    "scenario",
    "duplicate_decision",
    "artificial_decision",
    "effective_status",
    "reject_reason_codes",
    "review_stages",
]


with STATUS_FILE.open(
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
        current_status_rows
    )


# ==================================================
# Distribution
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


effective_status_counts = Counter(
    row[
        "effective_status"
    ]
    for row
    in current_status_rows
)


# Duplicate-stage KEEP ile
# artificial-stage reject aynı kayıt üzerinde
# bulunabilir. Bu çelişki değildir:
# KEEP yalnızca duplicate review kararıdır.
duplicate_keep_then_artificial_reject = sorted(
    sample_id
    for sample_id
    in artificial_reject_ids
    if (
        sample_id
        in duplicate_map
        and duplicate_map[
            sample_id
        ][
            "decision"
        ]
        == "KEEP"
    )
)


# ==================================================
# Summary
# ==================================================

summary = {
    "status":
        "STAGE7_CURRENT_REVIEW_STATE",

    "master_record_count":
        len(records),

    "master_sha256":
        sha256_file(
            MASTER_FILE
        ),

    "duplicate_ledger_sha256":
        sha256_file(
            DUPLICATE_LEDGER_FILE
        ),

    "artificial_decisions_sha256":
        sha256_file(
            ARTIFICIAL_DECISIONS_FILE
        ),

    "duplicate_reviewed_unique":
        len(
            duplicate_map
        ),

    "duplicate_keep_count":
        sum(
            1
            for row
            in duplicate_map.values()
            if row["decision"]
            == "KEEP"
        ),

    "duplicate_reject_count":
        len(
            duplicate_reject_ids
        ),

    "artificial_review_count":
        len(
            artificial_map
        ),

    "artificial_reject_count":
        len(
            artificial_reject_ids
        ),

    "duplicate_reject_artificial_reject_overlap":
        len(
            overlap_rejected
        ),

    "effective_provisional_reject_count":
        len(
            effective_reject_ids
        ),

    "current_survivor_count":
        len(
            survivors
        ),

    "effective_status_counts":
        dict(
            effective_status_counts
        ),

    "duplicate_keep_then_artificial_reject_count":
        len(
            duplicate_keep_then_artificial_reject
        ),

    "duplicate_keep_then_artificial_reject_sample_ids":
        duplicate_keep_then_artificial_reject,

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

    "master_dataset_modified":
        False,

    "survivor_is_final_dataset":
        False,

    "policy": (
        "Review decisions remain stage-specific. "
        "A duplicate-stage KEEP means only that the "
        "record was retained during duplicate review; "
        "it is not a final ACCEPT decision. Any "
        "provisional REJECT decision from a later QA "
        "stage removes the record from the current "
        "survivor view. Final ACCEPT/REJECT status "
        "will be assigned only after all Stage 7 "
        "reviews are complete."
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
print("STAGE 7D-1B - MERGED REVIEW STATE")
print("=" * 78)

print(
    f"Master kayıt               : "
    f"{len(records)}"
)

print("\nDuplicate review:")

print(
    f"  Reviewed unique          : "
    f"{len(duplicate_map)}"
)

print(
    f"  KEEP (stage-local)       : "
    f"{sum(1 for r in duplicate_map.values() if r['decision'] == 'KEEP')}"
)

print(
    f"  REJECT_DUPLICATE         : "
    f"{len(duplicate_reject_ids)}"
)


print("\nArtificial review:")

print(
    f"  Reviewed                 : "
    f"{len(artificial_map)}"
)

print(
    f"  REJECT_ARTIFICIAL        : "
    f"{len(artificial_reject_ids)}"
)

print(
    f"  Already duplicate-reject : "
    f"{len(overlap_rejected)}"
)


print("\nEffective state:")

print(
    f"  Provisional reject       : "
    f"{len(effective_reject_ids)}"
)

print(
    f"  Current survivor         : "
    f"{len(survivors)}"
)

print(
    f"  Dup-KEEP then artificial "
    f"reject                    : "
    f"{len(duplicate_keep_then_artificial_reject)}"
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
    f"  Stage ledger : "
    f"{STAGE_LEDGER_FILE}"
)

print(
    f"  Status       : "
    f"{STATUS_FILE}"
)

print(
    f"  Survivor     : "
    f"{SURVIVOR_FILE}"
)

print(
    f"  Summary      : "
    f"{SUMMARY_FILE}"
)