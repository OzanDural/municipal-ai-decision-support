import json
import re
from collections import Counter
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/master_stage2.jsonl"
)

FLAGS_FILE = Path(
    "ml/reports/dataset_qa/04_rule_flags.jsonl"
)

SUMMARY_FILE = Path(
    "ml/reports/dataset_qa/04_rule_summary.json"
)


# --------------------------------------------------
# Kurallar
# --------------------------------------------------

# Etiketi açıklayan / yapay biçimde dominant problemi
# okuyucuya söyleyen kalıplar.
#
# Bunlar otomatik REJECT değildir.
# Semantic review adayıdır.
# --------------------------------------------------

ARTIFICIAL_CONTRAST_PATTERNS = [
    r"\basıl sorun\b",
    r"\basıl problem\b",
    r"\basıl konu\b",
    r"\basıl mesele\b",
    r"\basıl korku(?:su|muz|ları)?\b",
    r"\besas sorun\b",
    r"\besas problem\b",
    r"\besas konu\b",
    r"\besas mesele\b",
    r"\besas acil\b",
    r"\böncelikli acil\b",
    r"\böncelik(?:li|le)?\b",
    r"\bbir kenara\b",
    r"\bönce .* müdahale\b",
    r"\bdaha önemli olan\b",
    r"\bönemli olan\b",
    r"\besas .{0,50} önceli",
    r"\bdikkat çekse de\b",
    r"\bmüdahalesinden önce\b",
]


# Doğrudan classification/routing dili.
META_LANGUAGE_PATTERNS = [
    r"\bkategori(?:ye|si|sine|sinde)?\b",
    r"\bsınıflandır(?:ma|ılması|ılmalı)?\b",
    r"\betiket(?:i|lenmesi|lemek)?\b",
    r"\blabel\b",
    r"\bclassification\b",
    r"\bbu şikayet .* kapsamına\b",
    r"\bbu şikâyet .* kapsamına\b",
    r"\bilgili kategori\b",
]


# Belediyedeki birim/hizmet ismi model için kolay
# label shortcut olabilir.
SERVICE_TERMS = [
    "zabıta",
    "temizlik işleri",
    "temizlik ekibi",
    "çevre koruma",
    "fen işleri",
    "yol ekibi",
    "park ve bahçeler",
    "park bahçeler",
    "park görevlileri",
    "veteriner işleri",
    "veteriner ekibi",
    "veterinerlik",
    "hayvan bakımevi",
    "rehabilitasyon merkezi",
    "çöp aracı",
    "çöp kamyonu",
]


# --------------------------------------------------
# PII benzeri örüntüler
# --------------------------------------------------

PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?90[\s.-]?)?"
    r"(?:0?5\d{2})"
    r"[\s.-]?\d{3}"
    r"[\s.-]?\d{2}"
    r"[\s.-]?\d{2}"
    r"(?!\d)"
)

TCKN_PATTERN = re.compile(
    r"(?<!\d)[1-9]\d{10}(?!\d)"
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+"
    r"\.[A-Za-z]{2,}\b"
)

BUILDING_NUMBER_PATTERN = re.compile(
    r"\b(?:no|no\.|numara|kapı no|daire)"
    r"\s*[:.]?\s*\d+\b",
    flags=re.IGNORECASE,
)


# --------------------------------------------------
# Yardımcı
# --------------------------------------------------

def normalize(text):
    return (
        text.casefold()
        .replace("â", "a")
    )


def add_flag(
    flags,
    record,
    flag_type,
    severity,
    detail,
):
    flags.append({
        "sample_id":
            record["sample_id"],

        "category":
            record["category"],

        "scenario":
            record["scenario"],

        "difficulty":
            record["difficulty"],

        "generation_batch":
            record["generation_batch"],

        "generator_id":
            record["generator_id"],

        "flag_type":
            flag_type,

        "severity":
            severity,

        "detail":
            detail,

        "complaint_text":
            record["complaint_text"],
    })


# --------------------------------------------------
# Oku
# --------------------------------------------------

records = []

with INPUT_FILE.open(
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
        f"bulunan {len(records)}"
    )


flags = []


# --------------------------------------------------
# Tarama
# --------------------------------------------------

for record in records:

    text = record[
        "complaint_text"
    ]

    normalized = normalize(
        text
    )


    # ----------------------------------------------
    # PII
    # ----------------------------------------------

    if PHONE_PATTERN.search(text):

        add_flag(
            flags,
            record,
            "PHONE_PATTERN",
            "HIGH",
            "Telefon numarası benzeri örüntü bulundu.",
        )


    if TCKN_PATTERN.search(text):

        add_flag(
            flags,
            record,
            "TCKN_PATTERN",
            "HIGH",
            "11 haneli TCKN benzeri örüntü bulundu.",
        )


    if EMAIL_PATTERN.search(text):

        add_flag(
            flags,
            record,
            "EMAIL_PATTERN",
            "HIGH",
            "E-posta adresi benzeri örüntü bulundu.",
        )


    if BUILDING_NUMBER_PATTERN.search(text):

        add_flag(
            flags,
            record,
            "BUILDING_NUMBER",
            "REVIEW",
            "Bina/kapı/daire numarası benzeri ifade bulundu.",
        )


    # ----------------------------------------------
    # Classification/meta leakage
    # ----------------------------------------------

    meta_hits = []

    for pattern in META_LANGUAGE_PATTERNS:

        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        ):

            meta_hits.append(
                pattern
            )


    if meta_hits:

        add_flag(
            flags,
            record,
            "META_LANGUAGE",
            "HIGH",
            {
                "matched_patterns":
                    meta_hits
            },
        )


    # ----------------------------------------------
    # Artificial contrast
    # ----------------------------------------------

    contrast_hits = []

    for pattern in (
        ARTIFICIAL_CONTRAST_PATTERNS
    ):

        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        ):

            contrast_hits.append(
                pattern
            )


    if contrast_hits:

        add_flag(
            flags,
            record,
            "ARTIFICIAL_CONTRAST",
            "REVIEW",
            {
                "matched_patterns":
                    contrast_hits
            },
        )


    # ----------------------------------------------
    # Service terms
    # ----------------------------------------------

    service_hits = []

    for term in SERVICE_TERMS:

        if term in normalized:

            service_hits.append(
                term
            )


    if service_hits:

        add_flag(
            flags,
            record,
            "SERVICE_TERM",
            "INFO",
            {
                "terms":
                    sorted(
                        set(service_hits)
                    )
            },
        )


# --------------------------------------------------
# Özet
# --------------------------------------------------

flag_type_counts = Counter(
    f["flag_type"]
    for f in flags
)

severity_counts = Counter(
    f["severity"]
    for f in flags
)

flagged_ids = {
    f["sample_id"]
    for f in flags
}


# Birden fazla rule flag alan kayıtlar
flags_per_id = Counter(
    f["sample_id"]
    for f in flags
)

multi_flag_records = {
    sample_id: count
    for sample_id, count
    in flags_per_id.items()
    if count > 1
}


summary = {
    "total_records":
        len(records),

    "total_flags":
        len(flags),

    "unique_flagged_records":
        len(flagged_ids),

    "flag_type_counts":
        dict(flag_type_counts),

    "severity_counts":
        dict(severity_counts),

    "multi_flag_record_count":
        len(multi_flag_records),

    "multi_flag_records":
        multi_flag_records,

    "policy": (
        "Rule-based flags are review signals, not "
        "automatic rejection decisions. HIGH and REVIEW "
        "records require semantic inspection. SERVICE_TERM "
        "is informational and is evaluated for potential "
        "label shortcut risk at dataset level."
    ),
}


FLAGS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with FLAGS_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for item in flags:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


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


# --------------------------------------------------
# Terminal
# --------------------------------------------------

print("=" * 78)
print("STAGE 4 - RULE-BASED QUALITY QA")
print("=" * 78)

print(
    f"Toplam kayıt          : {len(records)}"
)

print(
    f"Toplam flag           : {len(flags)}"
)

print(
    f"Flag alan kayıt       : {len(flagged_ids)}"
)


print("\nFlag türleri:")

for key, value in sorted(
    flag_type_counts.items()
):

    print(
        f"  {key:<25}: {value}"
    )


print("\nSeverity:")

for key, value in sorted(
    severity_counts.items()
):

    print(
        f"  {key:<10}: {value}"
    )


print(
    f"\nBirden fazla flag alan kayıt: "
    f"{len(multi_flag_records)}"
)


print("\nDosyalar:")

print(
    f"  Flags   : {FLAGS_FILE}"
)

print(
    f"  Summary : {SUMMARY_FILE}"
)