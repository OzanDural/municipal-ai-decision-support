import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CHALLENGE_FILE = Path("data/challenge/challenge_set_candidate_v1.jsonl")

EXPECTED_TOTAL = 150
EXPECTED_IN_SCOPE = 100
EXPECTED_OOD = 50
EXPECTED_IN_SCOPE_PER_CLASS = 20
EXPECTED_OOD_PER_FAMILY = 10

LABELS = {
    "TEMIZLIK_ATIK",
    "YOL_KALDIRIM",
    "PARK_YESIL_ALAN",
    "KENT_DUZENI_DENETIM",
    "HAYVAN_VETERINER",
}

EXPECTED_CHALLENGE_TYPES = {
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

if not CHALLENGE_FILE.exists():
    raise FileNotFoundError(f"Missing challenge file: {CHALLENGE_FILE}")

records = []
with CHALLENGE_FILE.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON at line {line_no}: {e}")
        records.append(obj)

if len(records) != EXPECTED_TOTAL:
    raise RuntimeError(f"Expected {EXPECTED_TOTAL}, found {len(records)}")

ids = [r.get("challenge_id") for r in records]
if len(ids) != len(set(ids)):
    raise RuntimeError("Duplicate challenge_id detected")

normalized_texts = [
    re.sub(r"\s+", " ", str(r.get("complaint_text", "")).strip().lower())
    for r in records
]
if any(not x for x in normalized_texts):
    raise RuntimeError("Empty complaint_text detected")
if len(normalized_texts) != len(set(normalized_texts)):
    raise RuntimeError("Exact normalized duplicate complaint detected")

groups = Counter(r.get("challenge_group") for r in records)
if groups["IN_SCOPE"] != EXPECTED_IN_SCOPE or groups["OOD"] != EXPECTED_OOD:
    raise RuntimeError(f"Unexpected group counts: {groups}")

in_scope = [r for r in records if r.get("challenge_group") == "IN_SCOPE"]
ood = [r for r in records if r.get("challenge_group") == "OOD"]

label_counts = Counter(r.get("gold_category") for r in in_scope)
if set(label_counts) != LABELS:
    raise RuntimeError(f"Unexpected in-scope labels: {label_counts}")
if any(label_counts[label] != EXPECTED_IN_SCOPE_PER_CLASS for label in LABELS):
    raise RuntimeError(f"Unbalanced in-scope labels: {label_counts}")

challenge_type_counts = Counter(r.get("challenge_type") for r in in_scope)
if set(challenge_type_counts) != EXPECTED_CHALLENGE_TYPES:
    raise RuntimeError(f"Unexpected challenge types: {challenge_type_counts}")
if any(challenge_type_counts[t] != 20 for t in EXPECTED_CHALLENGE_TYPES):
    raise RuntimeError(f"Unbalanced challenge types: {challenge_type_counts}")

# Stronger check: 4 examples of each challenge type inside every class.
cross = Counter((r.get("gold_category"), r.get("challenge_type")) for r in in_scope)
for label in LABELS:
    for ctype in EXPECTED_CHALLENGE_TYPES:
        if cross[(label, ctype)] != 4:
            raise RuntimeError(
                f"Expected 4 for ({label}, {ctype}), found {cross[(label, ctype)]}"
            )

if any(r.get("gold_category") is not None for r in ood):
    raise RuntimeError("OOD records must have gold_category=null")

ood_counts = Counter(r.get("ood_family") for r in ood)
if len(ood_counts) != 5:
    raise RuntimeError(f"Expected 5 OOD families, found {ood_counts}")
if any(v != EXPECTED_OOD_PER_FAMILY for v in ood_counts.values()):
    raise RuntimeError(f"Unbalanced OOD families: {ood_counts}")

bad_qa = [r["challenge_id"] for r in records if r.get("qa_status") != "PENDING_MANUAL_REVIEW"]
if bad_qa:
    raise RuntimeError(f"Unexpected qa_status: {bad_qa[:5]}")

print("=" * 80)
print("STAGE 14B - CHALLENGE SET STRUCTURAL VALIDATION")
print("=" * 80)
print(f"Total               : {len(records)}")
print(f"IN_SCOPE            : {groups['IN_SCOPE']}")
print(f"OOD                 : {groups['OOD']}")
print(f"Unique IDs          : {len(set(ids))}")
print(f"Normalized duplicates: 0")
print()
print("IN-SCOPE CATEGORY COUNTS")
for label in sorted(LABELS):
    print(f"  {label:<24}: {label_counts[label]}")
print()
print("IN-SCOPE CHALLENGE TYPE COUNTS")
for ctype in sorted(EXPECTED_CHALLENGE_TYPES):
    print(f"  {ctype:<24}: {challenge_type_counts[ctype]}")
print()
print("OOD FAMILY COUNTS")
for family, count in sorted(ood_counts.items()):
    print(f"  {family:<24}: {count}")
print()
print(f"SHA256              : {sha256_file(CHALLENGE_FILE)}")
print("Manual QA status    : PENDING")
print()
print("STATUS: CHALLENGE_SET_STRUCTURE_PASS")
