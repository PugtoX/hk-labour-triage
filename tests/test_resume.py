"""Regression: a subset run (`--categories X`) must NOT delete other categories' rows.

The file is rewritten from the in-memory rows at the end of main(), so anything
filtered out on the way in is destroyed. This test pins that down.

    python tests/test_resume.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("synth", ROOT / "synthesize.py")
synth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synth)

tmp = Path(tempfile.mkdtemp())
out = tmp / "synthetic.jsonl"

seed = [
    {"text": "工資問題一 這句夠長了吧", "category": "工資與扣薪",
     "urgency": "low", "escalate": False},
    {"text": "工資問題二 這句夠長了吧", "category": "工資與扣薪",
     "urgency": "high", "escalate": True},
    {"text": "假期問題一 這句夠長了吧", "category": "假期與休假",
     "urgency": "low", "escalate": False},
]
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in seed),
               encoding="utf-8")

synth.OUT = out
sys.argv = ["synthesize.py", "--categories", "假期與休假", "--per-category", "3"]

# the model returns nothing new — we only care that nothing is LOST
with mock.patch.object(synth, "call_ollama", return_value="[]"):
    synth.main()

after = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
         if line.strip()]
cats = sorted({r["category"] for r in after})

assert "工資與扣薪" in cats, f"工資與扣薪 rows were deleted by a subset run! cats={cats}"
assert "假期與休假" in cats, cats
assert len(after) >= len(seed), f"rows lost: {len(seed)} -> {len(after)}"
print(f"PASS subset run kept every category: {cats} ({len(after)} rows, was {len(seed)})")

print("\nRESUME SAFETY VERIFIED")
