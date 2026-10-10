#!/usr/bin/env python3
"""
Merge the LoRA adapter into the base model and emit a standalone model directory
plus an Ollama Modelfile.

    python export_model.py                    # -> models/triage-merged/
    python export_model.py --out /tmp/merged

Why: the adapter only runs through HF transformers, while the 9B baseline runs
through Ollama. That makes the latency comparison "framework + model" rather
than "model". Serving 1.7B through Ollama too removes that confound.

    ollama create triage-1.7b -f models/triage-merged/Modelfile
    python evaluate.py --model triage-1.7b --data data/test.jsonl --out v4_ollama_ft.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from schema import SYSTEM  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER_DIR = ROOT / "models" / "triage-lora"
DEFAULT_OUT = ROOT / "models" / "triage-merged"
MODEL_NAME = "triage-1.7b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--adapter", default=str(ADAPTER_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out)
    if not Path(args.adapter).exists():
        sys.exit(f"[!] adapter not found: {args.adapter} — train first")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[i] base={args.base}  adapter={args.adapter}")
    # `dtype`, not the deprecated `torch_dtype` (transformers 5.x warns on it).
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter)
    # merge_and_unload folds the LoRA deltas into the base weights, so the result
    # is a plain model with no peft dependency at inference time.
    model = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(args.base)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[i] merged model -> {out}")

    # The Modelfile has to carry the system prompt verbatim, and schema.py is the
    # single source of truth for it — so generate it rather than hand-copying.
    modelfile = out / "Modelfile"
    modelfile.write_text(
        f"FROM ./\n"
        f'SYSTEM """{SYSTEM}"""\n'
        f"PARAMETER temperature 0\n"
        f"PARAMETER num_predict 64\n"
        f"PARAMETER num_ctx 2048\n",
        encoding="utf-8",
    )
    print(f"[i] Modelfile -> {modelfile}")
    print(f"\n[next] ollama create {MODEL_NAME} -f {modelfile}")
    print(f"[next] python evaluate.py --model {MODEL_NAME} "
          f"--data data/test.jsonl --out v4_ollama_ft.json")


if __name__ == "__main__":
    main()
