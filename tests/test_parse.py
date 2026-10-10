"""Check synthesize.py's row parsing: the part most likely to be subtly wrong.

    python tests/test_parse.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("synth", ROOT / "synthesize.py")
synth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth)

parse = synth.parse_rows
REQ = "工資與扣薪"

cases: list[tuple[str, str, int]] = [
    ("bare array",
     '[{"text":"老闆拖我三個月工資，點算？","category":"工資與扣薪","urgency":"high","escalate":true}]',
     1),
    ("wrapped in an object",
     '{"items":[{"text":"年假未放完可以點？","category":"工資與扣薪","urgency":"medium","escalate":false}]}',
     1),
    ("single object, not a list",
     '{"text":"公司唔出糧係咪犯法？","category":"工資與扣薪","urgency":"medium","escalate":false}',
     1),
    ("string 'true' is coerced",
     '[{"text":"我老闆欠我兩萬蚊","category":"工資與扣薪","urgency":"high","escalate":"true"}]',
     1),
    ("bad urgency dropped",
     '[{"text":"這個問題夠長了吧","category":"工資與扣薪","urgency":"urgent","escalate":true}]',
     0),
    ("non-bool escalate dropped",
     '[{"text":"這個問題夠長了吧","category":"工資與扣薪","urgency":"high","escalate":"maybe"}]',
     0),
    ("unknown category dropped",
     '[{"text":"這個問題夠長了吧","category":"薪酬爭議","urgency":"high","escalate":true}]',
     0),
    ("missing category dropped",
     '[{"text":"這個問題夠長了吧","urgency":"high","escalate":true}]',
     0),
    ("too-short text dropped",
     '[{"text":"點算","category":"工資與扣薪","urgency":"high","escalate":true}]',
     0),
    ("invalid JSON yields nothing", "not json at all", 0),
    ("non-object entries dropped",
     '["just a string", {"text":"這個問題夠長了吧","category":"工資與扣薪","urgency":"low","escalate":false}]',
     1),
    # the generator may file a question under another category; that is the
    # self-check working, so the row is KEPT with its own label
    ("other category kept, not forced",
     '[{"text":"我今年有幾天年假？","category":"假期與休假","urgency":"low","escalate":false}]',
     1),
]

ok = True
for name, raw, want in cases:
    got = parse(raw, REQ)
    flag = "PASS" if len(got) == want else "FAIL"
    if len(got) != want:
        ok = False
    print(f"  {flag} {name}: got {len(got)}, want {want}")

# field-level correctness on the happy path
rows = parse('[{"text":"老闆拖我三個月工資，點算？","category":"工資與扣薪",'
             '"urgency":"high","escalate":true}]', REQ)
assert rows == [{"text": "老闆拖我三個月工資，點算？", "category": "工資與扣薪",
                 "urgency": "high", "escalate": True}], rows
print("  PASS field mapping")

# a row filed elsewhere must keep its own category, not the requested one
other = parse('[{"text":"我今年有幾天年假？","category":"假期與休假",'
              '"urgency":"low","escalate":false}]', REQ)
assert other[0]["category"] == "假期與休假", other
print("  PASS self-reported category is preserved")

assert len(synth.CATEGORY_NAMES) == 6, synth.CATEGORY_NAMES
assert set(synth.URGENCIES) == {"high", "medium", "low"}
print("  PASS category/urgency tables (6 categories)")

assert ok, "parse check failed"
print("\nPARSE VERIFIED")
