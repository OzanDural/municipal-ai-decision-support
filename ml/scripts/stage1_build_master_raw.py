import hashlib
import json
from collections import Counter
from pathlib import Path


SOURCE_DIR = Path(
    r"C:\Users\Lenovo\Desktop\belediye_dataset_raw"
)

OUTPUT_FILE = Path(
    "data/raw/master_raw.jsonl"
)

REPORT_FILE = Path(
    "ml/reports/dataset_qa/01_build_master_summary.json"
)


SOURCE_FILES = {
    "VeriGPT.txt": 500,
    "VeriGemini.txt": 500,
    "Veri_Claude.txt": 100,
    "UzunVeriGPT.txt": 300,
    "UzunVeriGemini.txt": 300,
    "UzunVeriClaude.txt": 50,
}

EXPECTED_TOTAL = sum(
    SOURCE_FILES.values()
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# SHA256
# --------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# --------------------------------------------------
# Markdown fence temizliği
#
# Model bazen çıktıyı ```json / ``` içine koymuş olabilir.
# JSON içeriğine dokunmuyoruz, sadece fence satırlarını atıyoruz.
# --------------------------------------------------

def remove_code_fences(text):

    lines = []

    for line in text.splitlines():

        stripped = line.strip()

        if stripped.startswith("```"):
            continue

        lines.append(line)

    return "\n".join(lines)


# --------------------------------------------------
# Bir dosyada art arda / yapışık JSON objelerini oku
# --------------------------------------------------

def decode_json_objects(path):

    raw_text = path.read_text(
        encoding="utf-8-sig"
    )

    raw_text = remove_code_fences(
        raw_text
    )

    decoder = json.JSONDecoder()

    objects = []

    position = 0
    length = len(raw_text)

    while position < length:

        # JSON objeleri arasındaki whitespace'i geç
        while (
            position < length
            and raw_text[position].isspace()
        ):
            position += 1

        if position >= length:
            break

        try:
            obj, end_position = decoder.raw_decode(
                raw_text,
                position,
            )

        except json.JSONDecodeError as e:

            context_start = max(
                0,
                position - 100,
            )

            context_end = min(
                length,
                position + 300,
            )

            context = raw_text[
                context_start:context_end
            ]

            raise ValueError(
                f"\nJSON decode hatası\n"
                f"Dosya   : {path}\n"
                f"Pozisyon: {position}\n"
                f"Hata     : {e}\n"
                f"Yakın içerik:\n{context}"
            ) from e

        if not isinstance(obj, dict):
            raise ValueError(
                f"{path} içinde JSON object olmayan "
                f"bir değer bulundu: {type(obj)}"
            )

        objects.append(obj)

        position = end_position

    return objects


# --------------------------------------------------
# Kaynak dosyaları oku
# --------------------------------------------------

all_records = []

source_summaries = {}


print("=" * 72)
print("STAGE 1 - BUILD MASTER RAW")
print("=" * 72)


for filename, expected_count in SOURCE_FILES.items():

    path = SOURCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Kaynak dosya bulunamadı:\n{path}"
        )

    records = decode_json_objects(
        path
    )

    actual_count = len(records)

    source_summaries[filename] = {
        "expected_count":
            expected_count,

        "actual_count":
            actual_count,

        "count_matches":
            actual_count == expected_count,

        "sha256":
            sha256_file(path),
    }

    print(
        f"{filename:<22} "
        f"beklenen={expected_count:<4} "
        f"bulunan={actual_count:<4} "
        f"{'OK' if actual_count == expected_count else 'HATA'}"
    )

    if actual_count != expected_count:

        raise ValueError(
            f"{filename} kayıt sayısı hatalı. "
            f"Beklenen={expected_count}, "
            f"Bulunan={actual_count}"
        )

    all_records.extend(
        records
    )


# --------------------------------------------------
# Toplam kayıt kontrolü
# --------------------------------------------------

if len(all_records) != EXPECTED_TOTAL:

    raise ValueError(
        f"Toplam kayıt sayısı uyuşmuyor. "
        f"Beklenen={EXPECTED_TOTAL}, "
        f"Bulunan={len(all_records)}"
    )


# --------------------------------------------------
# sample_id kontrolü
# --------------------------------------------------

sample_ids = [
    r.get("sample_id")
    for r in all_records
]

missing_sample_ids = [
    i
    for i, sample_id
    in enumerate(sample_ids, start=1)
    if not sample_id
]


if missing_sample_ids:

    raise ValueError(
        f"sample_id bulunmayan kayıtlar var: "
        f"{missing_sample_ids[:20]}"
    )


id_counts = Counter(
    sample_ids
)

duplicate_ids = [
    sample_id
    for sample_id, count
    in id_counts.items()
    if count > 1
]


if duplicate_ids:

    raise ValueError(
        f"Duplicate sample_id bulundu:\n"
        f"{duplicate_ids}"
    )


# --------------------------------------------------
# Master RAW yaz
# --------------------------------------------------

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    for record in all_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


master_sha256 = sha256_file(
    OUTPUT_FILE
)


# --------------------------------------------------
# Generator / batch ilk özet
# --------------------------------------------------

generator_distribution = Counter(
    r.get("generator_id")
    for r in all_records
)

batch_distribution = Counter(
    r.get("generation_batch")
    for r in all_records
)

category_distribution = Counter(
    r.get("category")
    for r in all_records
)


summary = {
    "expected_total":
        EXPECTED_TOTAL,

    "actual_total":
        len(all_records),

    "unique_sample_ids":
        len(set(sample_ids)),

    "duplicate_sample_id_count":
        len(duplicate_ids),

    "source_files":
        source_summaries,

    "generator_distribution":
        dict(generator_distribution),

    "batch_distribution":
        dict(batch_distribution),

    "category_distribution":
        dict(category_distribution),

    "master_raw_sha256":
        master_sha256,

    "master_raw_path":
        str(OUTPUT_FILE),
}


with REPORT_FILE.open(
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

print("\n" + "=" * 72)
print("MASTER RAW SONUCU")
print("=" * 72)

print(
    f"Toplam kayıt       : {len(all_records)}"
)

print(
    f"Unique sample_id   : {len(set(sample_ids))}"
)

print(
    f"Duplicate sample_id: {len(duplicate_ids)}"
)


print("\nGenerator dağılımı:")

for key, value in sorted(
    generator_distribution.items()
):
    print(
        f"  {key:<10} {value}"
    )


print("\nKategori dağılımı:")

for key, value in sorted(
    category_distribution.items()
):
    print(
        f"  {key:<25} {value}"
    )


print("\nSHA256:")

print(
    master_sha256
)


print("\nDosyalar:")

print(
    f"  Master : {OUTPUT_FILE}"
)

print(
    f"  Rapor  : {REPORT_FILE}"
)