"""Check check_dataset.split: nothing lost, no overlap, every category in every split.

    python tests/test_split.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("cds", ROOT / "check_dataset.py")
cds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cds)

# 6 categories x 20 rows
rows = []
for cat in cds.CATEGORY_NAMES:
    for i in range(20):
        rows.append({"text": f"{cat} 問題{i} 這句夠長了吧", "category": cat,
                     "urgency": cds.URGENCIES[i % 3],
                     "escalate": bool(i % 2)})

train, val, test = cds.split(rows, seed=42)
total = len(train) + len(val) + len(test)
assert total == len(rows), f"rows lost or duplicated: {total} vs {len(rows)}"
print(f"PASS no rows lost: {len(train)} + {len(val)} + {len(test)} = {total}")

tr = {r["text"] for r in train}
va = {r["text"] for r in val}
te = {r["text"] for r in test}
assert not (tr & va) and not (tr & te) and not (va & te), "splits overlap"
print("PASS no overlap between splits")

for part, name in ((train, "train"), (val, "val"), (test, "test")):
    missing = set(cds.CATEGORY_NAMES) - {r["category"] for r in part}
    assert not missing, f"{name} missing categories: {missing}"
print("PASS every category present in every split (stratified)")

t2, v2, e2 = cds.split(rows, seed=42)
assert [r["text"] for r in train] == [r["text"] for r in t2], "not reproducible"
assert [r["text"] for r in val] == [r["text"] for r in v2]
print("PASS same seed reproduces the same split")

t3, _, _ = cds.split(rows, seed=7)
assert [r["text"] for r in train] != [r["text"] for r in t3], "seed has no effect"
print("PASS a different seed changes the split")

small = [{"text": f"x{i} 這句夠長了吧", "category": "工資與扣薪",
          "urgency": "low", "escalate": False} for i in range(2)]
tr_s, va_s, te_s = cds.allocate(small, 0.15, 0.15)
assert len(tr_s) == 2 and not va_s and not te_s, (len(tr_s), len(va_s), len(te_s))
print("PASS tiny groups keep all rows in train")

uneven = ([{"text": f"a{i} 這句夠長了吧", "category": "工資與扣薪",
            "urgency": "low", "escalate": False} for i in range(34)]
          + [{"text": f"b{i} 這句夠長了吧", "category": "假期與休假",
              "urgency": "low", "escalate": False} for i in range(5)])
bal, trimmed = cds.balance(uneven, 20)
n_major = sum(1 for r in bal if r["category"] == "工資與扣薪")
n_minor = sum(1 for r in bal if r["category"] == "假期與休假")
assert n_major == 20, n_major
assert n_minor == 5, n_minor          # small categories must be left alone
assert trimmed == {"工資與扣薪": 14}, trimmed
print("PASS balance caps the majority category only")

print("\nSPLIT VERIFIED")
