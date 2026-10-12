import hashlib
import json
from pathlib import Path


INPUT_FILE = Path(
    "data/raw/master_raw.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

AUDIT_FILE = Path(
    "ml/reports/dataset_qa/02b_metadata_corrections.json"
)


# --------------------------------------------------
# Uzman incelemesi sonucu doğrulanmış metadata
# düzeltmeleri.
#
# Buraya yalnızca metnin kendisinin desteklediği
# düzeltmeler eklenir.
# Kota tutturmak amacıyla metadata değiştirilmez.
# --------------------------------------------------

CORRECTIONS = {
    "GEMINI_prod_gemini_001_001": {
        "language_style": {
            "from": "formal",
            "to": "neutral",
            "reason":
                "Metin doğal ve düz vatandaş dili kullanıyor; "
                "formal söylem özellikleri taşımıyor."
        }
    }
}


# --------------------------------------------------
# Bilinen fakat veriyi yapay biçimde değiştirmeden
# kabul edilen üretim sapmaları
# --------------------------------------------------

ACCEPTED_GENERATION_DEVIATIONS = [
    {
        "generation_batch":
            "long_gemini_002",

        "category":
            "HAYVAN_VETERINER",

        "field":
            "has_typo",

        "expected_true":
            4,

        "actual_true":
            3,

        "decision":
            "ACCEPTED_GENERATION_DEVIATION",

        "reason":
            (
                "Dördüncü typo örneğini oluşturmak için "
                "hatasız bir metnin metadata'sını yapay olarak "
                "has_typo=true yapmak metodolojik olarak uygun değildir."
            )
    }
]


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


# --------------------------------------------------
# Veriyi oku
# --------------------------------------------------

records = []

with INPUT_FILE.open(
    "r",
    encoding="utf-8"
) as f:

    for line_number, line in enumerate(
        f,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        records.append(
            json.loads(line)
        )


# --------------------------------------------------
# Düzeltmeleri uygula
# --------------------------------------------------

applied_corrections = []

found_ids = set()


for record in records:

    sample_id = record[
        "sample_id"
    ]

    if sample_id not in CORRECTIONS:
        continue

    found_ids.add(
        sample_id
    )

    correction_set = (
        CORRECTIONS[sample_id]
    )

    for field, correction in (
        correction_set.items()
    ):

        expected_old = (
            correction["from"]
        )

        new_value = (
            correction["to"]
        )

        actual_old = (
            record.get(field)
        )


        # Yanlış kayda / yanlış sürüme düzeltme
        # uygulanmasını engelle
        if actual_old != expected_old:

            raise ValueError(
                f"{sample_id}: "
                f"{field} beklenen eski değer "
                f"'{expected_old}', "
                f"fakat bulunan '{actual_old}'."
            )


        record[field] = (
            new_value
        )


        applied_corrections.append({
            "sample_id":
                sample_id,

            "field":
                field,

            "old_value":
                actual_old,

            "new_value":
                new_value,

            "reason":
                correction["reason"],
        })


# --------------------------------------------------
# Tüm correction ID'leri bulundu mu?
# --------------------------------------------------

missing_ids = (
    set(CORRECTIONS)
    - found_ids
)


if missing_ids:

    raise ValueError(
        "Düzeltme listesinde olup "
        f"dataset'te bulunmayan ID'ler: "
        f"{sorted(missing_ids)}"
    )


# --------------------------------------------------
# Yeni processed dataset
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
    newline="\n",
) as f:

    for record in records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


# --------------------------------------------------
# Audit raporu
# --------------------------------------------------

audit = {
    "input_file":
        str(INPUT_FILE),

    "output_file":
        str(OUTPUT_FILE),

    "input_record_count":
        len(records),

    "output_record_count":
        len(records),

    "correction_count":
        len(applied_corrections),

    "applied_corrections":
        applied_corrections,

    "accepted_generation_deviations":
        ACCEPTED_GENERATION_DEVIATIONS,

    "raw_sha256":
        sha256_file(INPUT_FILE),

    "processed_sha256":
        sha256_file(OUTPUT_FILE),

    "policy":
        (
            "RAW dataset is immutable. "
            "Only manually justified metadata corrections "
            "are applied to the processed dataset. "
            "Generation quota deviations are documented "
            "rather than artificially corrected."
        ),
}


AUDIT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with AUDIT_FILE.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit,
        f,
        ensure_ascii=False,
        indent=2,
    )


# --------------------------------------------------
# Terminal
# --------------------------------------------------

print("=" * 72)
print("STAGE 2B - METADATA CORRECTIONS")
print("=" * 72)

print(
    f"Girdi kayıt sayısı : {len(records)}"
)

print(
    f"Düzeltme sayısı    : {len(applied_corrections)}"
)

for item in applied_corrections:

    print(
        f"\n{item['sample_id']}"
    )

    print(
        f"  {item['field']}: "
        f"{item['old_value']} -> "
        f"{item['new_value']}"
    )


print(
    "\nKabul edilen üretim sapmaları:"
)

for deviation in (
    ACCEPTED_GENERATION_DEVIATIONS
):

    print(
        f"  "
        f"{deviation['generation_batch']} / "
        f"{deviation['category']} / "
        f"{deviation['field']}"
    )


print("\nRAW SHA256:")

print(
    audit["raw_sha256"]
)


print("\nPROCESSED SHA256:")

print(
    audit["processed_sha256"]
)


print("\nDosyalar:")

print(
    f"  Dataset : {OUTPUT_FILE}"
)

print(
    f"  Audit   : {AUDIT_FILE}"
)