import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CANDIDATE_FILE = Path("data/challenge/challenge_set_candidate_v1.jsonl")
MASTER_FINAL_FILE = Path("data/processed/master_final.jsonl")
FINAL_FILE = Path("data/challenge/challenge_set_final_v1.jsonl")
REPORT_DIR = Path("ml/reports/challenge")
LEDGER_FILE = REPORT_DIR / "14c_challenge_manual_qa_ledger.csv"
SUMMARY_FILE = REPORT_DIR / "14c_challenge_freeze_summary.json"

EXPECTED_CANDIDATE_SHA256 = "a1bbadb8cd88347460192fa76fa9a5f21bd85473dbc51b69e0dccfabab955d18"

CORRECTIONS = {'CH_IN_005': 'Park girişindeki bankların yanında konteyner taşmış durumda; çöpler yürüyüş yoluna kadar yayılmış, çocuklar oyun alanına giderken yanından geçiyor. Alanın temizlenmesini rica ediyorum.', 'CH_IN_006': 'Sahipsiz köpekler gece poşetleri parçalayınca atıklar bütün sokağa dağılıyor. Sabah yerde yiyecek artıkları ve ambalajlar kalıyor; konteynerin boşaltılıp çevrenin temizlenmesini istiyoruz.', 'CH_IN_017': 'Sokağın asfaltında küçük bozukluklar da var; son günlerde konteyner çevresinde alınmayan torbalar hızla birikiyor ve koku yapıyor. Birikmenin temizlenmesini istiyoruz.', 'CH_IN_018': 'Parkın ışıkları akşam biraz yetersiz kalıyor; girişteki atık yığını da her gün büyüyüp kötü koku yapıyor. Atıkların kaldırılıp alanın temizlenmesini rica ediyoruz.', 'CH_IN_020': 'Yakında sahipsiz hayvanlar dolaşıyor ve açıkta bırakılan yemek artıklarıyla ambalajlar sürekli etrafa saçılıyor. Atıkların düzenli toplanmasını ve alanın temizlenmesini rica ediyorum.', 'CH_IN_025': 'Parkın yanındaki kaldırımda büyük çatlak ve seviye farkı oluşmuş. Ağaçların arasından geçen bu bölümde yürürken insanlar takılıyor.', 'CH_IN_026': 'Çöp konteynerinin hemen önündeki asfalt çökmüş durumda. Araçlar konteynere yaklaşırken çökme olan bölüme sertçe giriyor ve manevra yapmak zorunda kalıyor.', 'CH_IN_027': 'Bir işletmenin önündeki kaldırım taşları kırılmış ve yerinden çıkmış. Yayalar dükkânın önünden geçerken taşlara takılıyor ve yüzeyin onarılması gerekiyor.', 'CH_IN_028': 'Sahipsiz hayvanların sık geçtiği ara sokakta yol yüzeyi ciddi şekilde bozulmuş; araçlar çökme olan bölümden kaçmak için karşı şeride taşıyor.', 'CH_IN_037': 'Sokakta birkaç çöp poşeti de bırakılmış; araçlar ise yolun ortasındaki büyük çökmeden kaçmak için sürekli manevra yapıyor. Yüzeyin onarılmasını rica ediyoruz.', 'CH_IN_038': 'Park girişinde sulama da aksıyor; girişe ulaşan kaldırımın yarısı kırılmış ve bebek arabasıyla geçilemiyor. Kırık bölümün onarılmasını rica ederim.', 'CH_IN_039': 'Bir dükkânın tabelası biraz dışarı taşıyor; hemen önündeki bordür kopmuş ve parçaları yola düşmüş. Taşların onarılıp sabitlenmesini istiyoruz.', 'CH_IN_040': 'Gece sokakta köpekler de oluyor; yolun ortasındaki derin çökme nedeniyle sürücüler son anda sert manevra yapıyor. Çukurun kazaya neden olmadan onarılmasını rica ederim.', 'CH_IN_045': 'Parkın girişinde bir konteyner de bulunuyor; çocuk oyun grubundaki kırık basamak ve açıkta kalan metal kenar kullanım sırasında tehlike oluşturuyor.', 'CH_IN_046': 'Park içinde birkaç sahipsiz kedi dolaşıyor; aynı bölgede sulama çalışmadığı için çimler hızla kuruyor. Sulama sisteminin kontrol edilmesini rica ederim.', 'CH_IN_048': 'Yakındaki işletmeden zaman zaman ses geliyor; park içindeki kuru ve çatlamış büyük dal oyun alanının üzerine kadar sarkmış. Dalın kontrol edilmesini istiyoruz.', 'CH_IN_057': 'Park girişinde yerde birkaç kâğıt parçası var; çocukların kullandığı salıncağın bağlantısı da gevşemiş ve kopacak gibi duruyor. Oyun ekipmanının kontrol edilmesini rica ederim.', 'CH_IN_058': 'Yakındaki kaldırım biraz bozuk; park içindeki iki bankın tahtaları kırılmış ve sivri parçalar açıkta kalmış. Bankların onarılmasını istiyoruz.', 'CH_IN_059': 'Akşam çevrede köpekler görülebiliyor; oyun alanının üzerindeki kalın ağaç dalı da çatlamış ve aşağı doğru sarkmış. Dalın düşmeden budanmasını rica ederim.', 'CH_IN_065': 'Kafenin önünde birkaç karton kutu da duruyor; masa ve sandalyeler kaldırımın tamamına yayıldığı için yayalar yola inmek zorunda kalıyor. Denetim yapılmasını istiyoruz.', 'CH_IN_067': 'Park girişine yakın seyyar tezgâh yaya yolunun ortasına kuruluyor. Tezgâhın çevresinden geçemeyenler taşıt yoluna iniyor.', 'CH_IN_068': 'Mağaza ürünleri sağlam kaldırım taşlarının üzerine sıra sıra çıkarılmış ve geçiş alanı neredeyse tamamen kapanmış. Ürünlerin geri çekilmesi için denetim yapılmasını istiyoruz.', 'CH_IN_077': 'Kaldırımda küçük bir çatlak var; restoranın dışarı çıkardığı masa ve sandalyeler geçiş alanını boydan boya kaplıyor. Alanın açılması için denetim istiyoruz.', 'CH_IN_078': 'Tezgâhın yanında biraz ambalaj birikmiş; satış standı yaya geçişini tamamen kapatıyor ve insanlar yola iniyor. Seyyar satışın kontrol edilmesini rica ederim.', 'CH_IN_079': 'Yakında birkaç sahipsiz hayvan dolaşıyor; gece yarısına kadar devam eden işletme müziği evlerin içine kadar geliyor. Gürültü denetimi yapılmasını rica ederiz.', 'CH_IN_080': 'Mağaza önündeki bordürde hafif hasar var; dışarı çıkarılmış ürünler ve reklam panoları kalan geçiş alanını neredeyse tamamen kapatıyor. Denetim yapılmasını istiyoruz.', 'CH_IN_085': 'Konteyner çevresinde çöpler birikmiş; hemen yanında bir köpek hareketsiz yatıyor ve nefes almakta zorlanıyor. Veteriner ekibi yönlendirilmesini rica ediyorum.', 'CH_IN_086': 'Parkta kırık bir bank bulunuyor; yakınındaki çalılıkta ön ayağından yaralı bir kedi saklanıyor ve yürümekte zorlanıyor. Veteriner ekibinin almasını rica ederim.', 'CH_IN_097': 'Konteyner dolmaya başlamış; yanında araç çarpmış gibi yerde yatan köpek kıpırdamıyor. Hayvanın veteriner ekiplerince alınmasını rica ediyorum.', 'CH_IN_098': 'Parkın sulaması aksıyor; çalılıkların içinde ciddi şekilde yaralanmış bir kedi de uzun süredir kıpırdamadan yatıyor. Veteriner ekibi yönlendirilmesini rica ederim.', 'CH_IN_099': 'Kaldırım taşlarından biri kırık; okul önünde dolaşan köpek grubu çocuklara doğru koşup havlıyor ve geçişi korkutucu hale getiriyor. Ekip yönlendirilmesini istiyoruz.'}

LABELS = {
    "TEMIZLIK_ATIK",
    "YOL_KALDIRIM",
    "PARK_YESIL_ALAN",
    "KENT_DUZENI_DENETIM",
    "HAYVAN_VETERINER",
}

CHALLENGE_TYPES = {
    "INDIRECT_WORDING",
    "CROSS_CLASS_DISTRACTOR",
    "TYPO_NOISE",
    "LEXICAL_SHORTCUT",
    "MULTI_ISSUE_DOMINANT",
}

def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {line_no}: {e}")
    return records

def norm(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(f"Missing candidate challenge file: {CANDIDATE_FILE}")

candidate_hash = sha256_file(CANDIDATE_FILE)
if candidate_hash != EXPECTED_CANDIDATE_SHA256:
    raise RuntimeError(
        "Candidate SHA256 mismatch.\n"
        f"Expected: {EXPECTED_CANDIDATE_SHA256}\n"
        f"Found   : {candidate_hash}"
    )

records = load_jsonl(CANDIDATE_FILE)
if len(records) != 150:
    raise RuntimeError(f"Expected 150 records, found {len(records)}")

ids = [r["challenge_id"] for r in records]
if len(ids) != len(set(ids)):
    raise RuntimeError("Duplicate challenge_id detected")

ledger_rows = []
for r in records:
    original = r["complaint_text"]
    revised = CORRECTIONS.get(r["challenge_id"], original)
    changed = revised != original
    r["complaint_text"] = revised
    r["qa_status"] = "MANUAL_REVIEW_PASS"

    ledger_rows.append({
        "challenge_id": r["challenge_id"],
        "challenge_group": r["challenge_group"],
        "gold_category": r.get("gold_category"),
        "ood_family": r.get("ood_family"),
        "challenge_type": r.get("challenge_type"),
        "decision": "PASS",
        "text_revised": changed,
        "label_changed": False,
        "ood_family_changed": False,
        "review_note": (
            "Wording revised to remove artificial contrast/target-explaining language; semantic target unchanged."
            if changed else
            "Label/OOD status and wording reviewed; no change required."
        ),
    })

in_scope = [r for r in records if r["challenge_group"] == "IN_SCOPE"]
ood = [r for r in records if r["challenge_group"] == "OOD"]

if len(in_scope) != 100 or len(ood) != 50:
    raise RuntimeError("Unexpected IN_SCOPE/OOD counts")

label_counts = Counter(r["gold_category"] for r in in_scope)
if set(label_counts) != LABELS or any(v != 20 for v in label_counts.values()):
    raise RuntimeError(f"Unexpected class distribution: {label_counts}")

type_counts = Counter(r["challenge_type"] for r in in_scope)
if set(type_counts) != CHALLENGE_TYPES or any(v != 20 for v in type_counts.values()):
    raise RuntimeError(f"Unexpected challenge type distribution: {type_counts}")

cross = Counter((r["gold_category"], r["challenge_type"]) for r in in_scope)
for label in LABELS:
    for challenge_type in CHALLENGE_TYPES:
        if cross[(label, challenge_type)] != 4:
            raise RuntimeError(
                f"Expected 4 for {label} / {challenge_type}, "
                f"found {cross[(label, challenge_type)]}"
            )

ood_counts = Counter(r["ood_family"] for r in ood)
if len(ood_counts) != 5 or any(v != 10 for v in ood_counts.values()):
    raise RuntimeError(f"Unexpected OOD distribution: {ood_counts}")

if any(r.get("gold_category") is not None for r in ood):
    raise RuntimeError("OOD records must have gold_category=null")

normalized = [norm(r["complaint_text"]) for r in records]
if len(normalized) != len(set(normalized)):
    raise RuntimeError("Normalized duplicate exists inside challenge set")

# Ensure no exact/normalized duplicate with the frozen accepted dataset.
master_overlap = 0
if MASTER_FINAL_FILE.exists():
    master = load_jsonl(MASTER_FINAL_FILE)
    master_texts = {norm(r["complaint_text"]) for r in master}
    master_overlap = sum(1 for r in records if norm(r["complaint_text"]) in master_texts)
    if master_overlap != 0:
        raise RuntimeError(
            f"Challenge/master_final normalized overlap detected: {master_overlap}"
        )

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_FILE.parent.mkdir(parents=True, exist_ok=True)

with FINAL_FILE.open("w", encoding="utf-8", newline="\n") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with LEDGER_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=ledger_rows[0].keys())
    writer.writeheader()
    writer.writerows(ledger_rows)

final_hash = sha256_file(FINAL_FILE)

summary = {
    "status": "CHALLENGE_SET_V1_FROZEN",
    "candidate_sha256": candidate_hash,
    "final_sha256": final_hash,
    "total_records": len(records),
    "in_scope_records": len(in_scope),
    "ood_records": len(ood),
    "manual_reviewed_records": len(records),
    "manual_pass": len(records),
    "manual_reject": 0,
    "text_revisions": len(CORRECTIONS),
    "gold_category_changes": 0,
    "ood_family_changes": 0,
    "normalized_overlap_with_master_final": master_overlap,
    "model_predictions_seen_before_freeze": False,
    "review_policy_already_frozen": True,
    "purpose": (
        "Diagnostic challenge evaluation only; not training data and "
        "not a replacement for the frozen standard test."
    ),
}

with SUMMARY_FILE.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("=" * 80)
print("STAGE 14C - CHALLENGE MANUAL QA + FREEZE")
print("=" * 80)
print(f"Total reviewed      : {len(records)}")
print(f"Manual PASS         : {len(records)}")
print("Manual REJECT       : 0")
print(f"Text revisions      : {len(CORRECTIONS)}")
print("Gold label changes  : 0")
print("OOD family changes  : 0")
print(f"Master overlap      : {master_overlap}")
print()
print(f"IN_SCOPE            : {len(in_scope)}")
print(f"OOD                 : {len(ood)}")
print()
print(f"Candidate SHA256    : {candidate_hash}")
print(f"FINAL SHA256        : {final_hash}")
print()
print("Model predictions used for QA: False")
print("Review policy frozen before challenge: True")
print()
print(f"Final challenge set : {FINAL_FILE}")
print(f"QA ledger           : {LEDGER_FILE}")
print(f"Summary             : {SUMMARY_FILE}")
print()
print("STATUS: CHALLENGE_SET_V1_FROZEN")
