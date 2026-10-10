"""Verify evaluate.py: prediction parsing and test-set loading.

    python tests/test_evaluate.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from schema import to_messages  # noqa: E402

spec = importlib.util.spec_from_file_location("ev", ROOT / "evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

parse = ev.parse_prediction
GOOD = '{"category":"工資與扣薪","urgency":"high","escalate":true}'

cases = [
    ("plain object", GOOD, True),
    ("array wrapped", f"[{GOOD}]", True),
    ("string bool coerced",
     '{"category":"工資與扣薪","urgency":"high","escalate":"true"}', True),
    # escalate=False is a VALID label but a falsy value — this case was missing,
    # and its absence hid a bug that made the whole baseline read 16.7%.
    ("escalate false is valid",
     '{"category":"工資與扣薪","urgency":"medium","escalate":false}', True),
    ("string 'false' coerced",
     '{"category":"工資與扣薪","urgency":"low","escalate":"false"}', True),
    ("missing field", '{"category":"工資與扣薪","urgency":"high"}', False),
    ("empty category", '{"category":"","urgency":"high","escalate":true}', False),
    ("empty urgency", '{"category":"工資與扣薪","urgency":"","escalate":true}', False),
    ("bad escalate", '{"category":"工資與扣薪","urgency":"high","escalate":"maybe"}', False),
    ("invalid json", "not json at all", False),
    ("empty array", "[]", False),
    ("not an object", '"just a string"', False),
]
ok = True
for name, raw, want_ok in cases:
    got = parse(raw)
    good = (got is not None) == want_ok
    ok = ok and good
    print(f"  {'PASS' if good else 'FAIL'} {name}: -> {got}")

assert parse(GOOD) == {"category": "工資與扣薪", "urgency": "high", "escalate": True}
print("  PASS field values preserved")

tmp = Path(tempfile.mkdtemp()) / "test.jsonl"
row = {"text": "公司拖我工資", "category": "工資與扣薪", "urgency": "high", "escalate": True}
tmp.write_text(json.dumps(to_messages(row), ensure_ascii=False), encoding="utf-8")
loaded = ev.load_test(tmp)
assert loaded == [{"question": "公司拖我工資",
                   "gold": {"category": "工資與扣薪", "urgency": "high",
                            "escalate": True}}], loaded
print("  PASS load_test reads the chat-format test set")

assert ok, "parse cases failed"
print("\nEVALUATE VERIFIED")
