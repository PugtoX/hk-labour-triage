#!/usr/bin/env python3
"""
QLoRA fine-tuning for the triage task (peft + transformers, no unsloth).

    python train.py --smoke        # 2 optimizer steps: does it fit, does loss move?
    python train.py                # full run
    python train.py --eval-only    # evaluate the saved adapter on the test set

The SYSTEM prompt comes from schema.py — the same one evaluate.py uses. Training
on one contract and judging against another is a silent failure, so it is
imported, never copied.
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from schema import CATEGORY_NAMES, FIELDS, SYSTEM, URGENCIES  # noqa: E402
from evaluate import parse_prediction  # noqa: E402  (one parser, one json_valid)

BASE_MODEL = "Qwen/Qwen3-1.7B"  # standard Qwen3ForCausalLM; fp16 fits in 8GB
ADAPTER_DIR = ROOT / "models" / "triage-lora"
MAX_LEN = 1536      # the SYSTEM prompt alone is ~850 chars of Chinese
SEED = 42


def load_records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line)["messages"])
    return rows


def build_tokenizer(base: str):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token          # needed for batching
    return tok


def load_base(base: str, quant: bool = True):
    """Load the base model.

    quant=True  -> 4-bit QLoRA (bnb). Cheaper for 7B+, but the backward pass
                   materialises an fp16 copy of every quantised weight, so for a
                   1.7B model it costs MORE VRAM than fp16 (measured: 7.7GB vs
                   ~4.6GB) and runs ~40x slower when that overflows.
    quant=False -> fp16 LoRA, no dequantisation buffers.
    """
    if quant:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map={"": 0})

    model.config.use_cache = False
    # Gradient checkpointing is required (activations at ~700 tokens overflow an
    # 8GB card), and the DEFAULT reentrant path is drastically slower with PEFT
    # LoRA. PyTorch 2.9+ drops the implicit default anyway, so pass it explicitly.
    if quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()   # let gradients reach the LoRA inputs
    print(f"[i] gradient checkpointing enabled: "
          f"{getattr(model, 'is_gradient_checkpointing', '?')}")
    # A silent CPU/disk offload is the classic cause of "it runs but is 30x
    # slower than it should be" — print where the layers actually landed.
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        devices = sorted({str(v) for v in device_map.values()})
        print(f"[i] hf_device_map devices: {devices}")
        if any(("cpu" in d or "disk" in d) for d in devices):
            print("[!] some layers are on CPU/disk — that is ~100x slower")
    return model


def attach_lora(model):
    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def render(tok, messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Apply the chat template, forcing thinking OFF.

    Qwen3 can emit a reasoning trace before the answer. Our training targets are
    a bare JSON object, so leaving thinking on teaches the model to produce
     thinking blocks instead of labels. The flag is only passed when the template
    actually advertises it — swallowing TypeError here would also hide genuine
    template errors.
    """
    kw: dict = {}
    if add_generation_prompt:
        kw["add_generation_prompt"] = True
    if "enable_thinking" in (getattr(tok, "chat_template", None) or ""):
        kw["enable_thinking"] = False
    return tok.apply_chat_template(messages, tokenize=False, **kw)


def encode(tok, messages: list[dict]) -> dict | None:
    """Tokens for the whole conversation, with the prompt part masked out of the
    loss — the model must learn to PRODUCE the label, not to reproduce the
    system prompt.

    Returns None when truncation leaves nothing but the prompt: every label
    would be -100, the loss would be meaningless, and training would silently
    learn nothing.
    """
    full = render(tok, messages)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    prefix = render(tok, messages[:-1], add_generation_prompt=True)
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    # The masking below ASSUMES the generation prompt is a literal prefix of the
    # full conversation. If a template ever breaks that, the labels would be
    # misaligned and the model would be trained on the wrong targets — silently.
    if ids[:len(prefix_ids)] != prefix_ids:
        return None
    ids = ids[:MAX_LEN]
    n_mask = min(len(prefix_ids), len(ids))
    if n_mask >= len(ids):
        return None
    labels = [-100] * n_mask + ids[n_mask:]
    return {"input_ids": ids, "labels": labels,
            "attention_mask": [1] * len(ids)}


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def mean_loss(model, samples: list[dict], pad_id: int) -> float:
    """Held-out loss. Train loss alone cannot show overfitting — it always looks
    good once the model memorises. The gap between the two is the signal."""
    if not samples:
        return float("nan")
    model.eval()
    total = 0.0
    with torch.no_grad():
        for s in samples:
            batch = {k: v.to(model.device)
                     for k, v in collate([s], pad_id).items()}
            total += model(**batch).loss.item()
    model.train()
    return total / len(samples)


def evaluate(model, tok, rows: list[dict], max_new: int = 64,
             verbose: bool = False, out_path: str = "") -> dict:
    """Greedy decode and score — the same metrics evaluate.py reports.

    With out_path, the per-item gold/pred is written out. Without it the
    predictions are lost, and questions like "does a third model agree with the
    human on row 7?" cannot be answered without re-running everything.
    """
    model.eval()
    # use_cache is off during training (gradient checkpointing needs it off), but
    # greedy decoding without a KV cache is 2-5x slower — the reported latency
    # would not be comparable to the baseline.
    prev_cache = model.config.use_cache
    model.config.use_cache = True
    correct = {f: 0 for f in FIELDS}
    exact = json_ok = 0
    preds: list[dict | None] = []
    gen_lens: list[int] = []
    details: list[dict] = []
    t0 = time.perf_counter()
    for messages in rows:
        prompt = render(tok, messages[:-1], add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        gen_lens.append(int(new_tokens.shape[0]))
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        # One shared parser (evaluate.parse_prediction): it guards against a list
        # or bare string coming back, which `json.loads` accepts happily and then
        # `pred.get(f)` blows up on — killing the whole run instead of scoring
        # that row as wrong. json_valid also now means the same thing here as in
        # evaluate.py.
        try:
            pred = parse_prediction(text)
        except Exception as e:
            print(f"    [!] parser blew up on {text[:80]!r}: {type(e).__name__}: {e}")
            pred = None
        json_ok += pred is not None
        preds.append(pred)
        gold = json.loads(messages[-1]["content"])
        hit = {f: bool(pred and pred.get(f) == gold[f]) for f in FIELDS}
        for f in FIELDS:
            correct[f] += hit[f]
        exact += all(hit.values())
        details.append({"question": messages[-2]["content"], "gold": gold,
                        "pred": pred, "hit": hit, "gen_tokens": gen_lens[-1]})
    n = len(rows)
    dt = (time.perf_counter() - t0) / n * 1000
    model.config.use_cache = prev_cache
    model.train()

    if verbose:
        # Collapse detector. A constant predictor scores exactly the majority
        # class share, so accuracy alone cannot tell "collapsed to one value"
        # from "genuinely close". The distributions can.
        for f in FIELDS:
            gold_dist = Counter(json.loads(m[-1]["content"])[f] for m in rows)
            pred_dist = Counter(p[f] for p in preds if p)
            majority = max(gold_dist.values()) / n
            acc = correct[f] / n
            flags = []
            if len(pred_dist) == 1 and len(gold_dist) > 1:
                flags.append("COLLAPSED to a single value")
            if acc <= majority + 0.02:
                flags.append(f"no better than majority ({majority:.1%})")
            note = ("   [!] " + "; ".join(flags)) if flags else ""
            print(f"    [dist] {f}: acc={acc:.1%} majority={majority:.1%} "
                  f"pred={dict(pred_dist)}{note}")
        # Generation length: if it always maxes out, the model never learned to
        # stop (which the latency number would then be measuring).
        at_cap = sum(1 for g in gen_lens if g >= max_new)
        print(f"    [gen] tokens: min={min(gen_lens)} median="
              f"{sorted(gen_lens)[len(gen_lens) // 2]} max={max(gen_lens)} "
              f"(cap {max_new}); {at_cap}/{n} hit the cap")

        # Per-class recall + confusion matrix for urgency. Overall accuracy hides
        # the case that matters most: a model that shoves everything into
        # "medium" scores respectably while missing every genuinely urgent item.
        cm = Counter((json.loads(m[-1]["content"])["urgency"], p["urgency"])
                     for m, p in zip(rows, preds) if p)
        for g in URGENCIES:
            total = sum(v for (gg, _), v in cm.items() if gg == g)
            print(f"    [recall] urgency={g:<7} {cm[(g, g)]}/{total} = "
                  f"{(cm[(g, g)] / total if total else 0.0):.1%}")
        print("    [cm] urgency gold->pred: "
              + str({f"{g}->{p}": v for (g, p), v in sorted(cm.items())}))

    report = {**{f"{f}_acc": correct[f] / n for f in FIELDS},
              "exact_match": exact / n, "json_valid": json_ok / n,
              "latency_ms_per_item": round(dt)}
    if out_path:
        Path(out_path).write_text(
            json.dumps({"report": report, "details": details},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    [out] per-item gold/pred -> {out_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="2 optimizer steps only")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-adapter", action="store_true",
                    help="with --eval-only: score the BASE model, no LoRA adapter. "
                         "The control that separates 'fine-tuning broke the field' "
                         "from 'this inference path never could do it'.")
    ap.add_argument("--no-quant", action="store_true",
                    help="fp16 LoRA instead of 4-bit QLoRA (less VRAM for <3B)")
    ap.add_argument("--out", default="",
                    help="write per-item gold/pred JSON here — needed to compare "
                         "two models row by row (e.g. to settle a disputed label)")
    args = ap.parse_args()

    train_rows = load_records(ROOT / "data" / "train.jsonl")
    val_rows = load_records(ROOT / "data" / "val.jsonl")
    test_rows = load_records(ROOT / "data" / "test.jsonl")
    print(f"[i] train={len(train_rows)}  val={len(val_rows)}  "
          f"test={len(test_rows)}  base={args.base}")

    # Contract self-check. The system message carried INSIDE the data is what the
    # model is trained on and judged with: evaluate() renders messages[:-1], i.e.
    # system + user. If it has drifted from schema.SYSTEM then every number is
    # comparing two different contracts, so fail loudly instead of silently.
    for name, rows in (("train", train_rows), ("test", test_rows)):
        if rows and rows[0] and rows[0][0].get("role") == "system":
            if rows[0][0]["content"] != SYSTEM:
                sys.exit(f"[!] {name}.jsonl carries a stale system prompt "
                         f"({len(rows[0][0]['content'])} chars; schema has "
                         f"{len(SYSTEM)}) — re-run check_dataset.py")
            print(f"[i] {name} system prompt matches schema.SYSTEM "
                  f"({len(SYSTEM)} chars) — contract OK")
    print(f"[i] train={len(train_rows)}  val={len(val_rows)}  "
          f"test={len(test_rows)}  base={args.base}")

    tok = build_tokenizer(args.base)

    if args.eval_only:
        model = load_base(args.base, quant=not args.no_quant)
        if args.no_adapter:
            print("[i] BASE model only (no adapter) — the control run")
        else:
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
            print("[i] adapter loaded from", ADAPTER_DIR)
        print("[i] test:", json.dumps(
            evaluate(model, tok, test_rows, verbose=True, out_path=args.out),
            indent=2))
        return

    model = attach_lora(load_base(args.base, quant=not args.no_quant))
    encoded, skipped = [], 0
    for m in train_rows:
        e = encode(tok, m)
        if e is None:
            skipped += 1
        else:
            encoded.append(e)
    val_encoded = [e for e in (encode(tok, m) for m in val_rows) if e]

    if not encoded:
        # Measure instead of guessing: report how long the prompt actually is,
        # so MAX_LEN can be set from evidence.
        probes = []
        for m in train_rows[:5]:
            prefix = render(tok, m[:-1], add_generation_prompt=True)
            full = render(tok, m)
            probes.append((len(tok(prefix, add_special_tokens=False)["input_ids"]),
                           len(tok(full, add_special_tokens=False)["input_ids"])))
        print("[!] every row was skipped — prompt vs full token counts:")
        for p, f in probes:
            print(f"      prompt={p}  full={f}")
        need = max(f for _, f in probes) + 8
        sys.exit(f"[!] MAX_LEN={MAX_LEN} is too small; set it to at least {need}")

    if skipped:
        print(f"[!] skipped {skipped} rows: the prompt alone already hit MAX_LEN")
    lens = sorted(len(e["input_ids"]) for e in encoded)
    print(f"[i] token lengths: min={lens[0]} median={lens[len(lens) // 2]} "
          f"max={lens[-1]} (MAX_LEN={MAX_LEN})")

    rng = random.Random(SEED)
    order = list(range(len(encoded)))
    steps_per_epoch = max(1, len(encoded) // args.accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.smoke:
        total_steps = 2
        print("[smoke] 2 optimizer steps only")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    print(f"[i] steps/epoch={steps_per_epoch}  total_steps={total_steps}")
    print(f"[i] VRAM before training: "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    # Time one forward and one forward+backward. If the forward ALONE is slow,
    # the weights are being moved (offload), not computed. Run it in train mode
    # so dropout matches the real loop.
    model.train()
    probe = collate([encoded[0]], tok.pad_token_id)
    probe = {k: v.to(model.device) for k, v in probe.items()}
    t = time.perf_counter()
    with torch.no_grad():
        model(**probe)
    fwd = time.perf_counter() - t
    t = time.perf_counter()
    model(**probe).loss.backward()
    model.zero_grad(set_to_none=True)
    bwd = time.perf_counter() - t
    print(f"[diag] one sample ({probe['input_ids'].shape[1]} tokens): "
          f"forward={fwd:.2f}s  forward+backward={bwd:.2f}s")

    model.train()
    micro = 0
    step = 0
    best_val = -1.0
    t0 = time.perf_counter()
    while step < total_steps:
        rng.shuffle(order)
        for idx in order:
            if step >= total_steps:
                break
            batch = collate([encoded[idx]], tok.pad_token_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                peak = torch.cuda.max_memory_allocated()
                cap = torch.cuda.get_device_properties(0).total_memory
                warn = ("   [!] >90% VRAM — this will page to RAM and crawl"
                        if peak > 0.9 * cap else "")
                print(f"  step {step}/{total_steps}  "
                      f"loss={loss.item() * args.accum:.4f}  "
                      f"peak_vram={peak / 1e9:.2f}GB / {cap / 1e9:.1f}GB{warn}")
                # Held-out score at each epoch boundary. val LOSS is nearly
                # useless here: ~75% of the label tokens are the fixed JSON
                # skeleton, so a model that memorises the skeleton alone scores
                # a tiny loss while getting every field wrong. exact_match is
                # the metric that matters, so it drives best-checkpoint selection.
                if val_encoded and (step % steps_per_epoch == 0
                                    or step == total_steps):
                    tr_loss = loss.item() * args.accum
                    vl = mean_loss(model, val_encoded, tok.pad_token_id)
                    vm = evaluate(model, tok, val_rows)["exact_match"]
                    print(f"    [val] step {step}: val_exact_match={vm:.1%}  "
                          f"val_loss={vl:.4f}  (train_loss={tr_loss:.4f})")
                    if not args.smoke and vm > best_val:
                        best_val = vm
                        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(str(ADAPTER_DIR))
                        tok.save_pretrained(str(ADAPTER_DIR))
                        print(f"      [best] saved — val_exact_match={vm:.1%}")

    elapsed = time.perf_counter() - t0
    micro_batches = max(1, step * args.accum)
    print(f"[i] training took {elapsed:.1f}s  "
          f"({elapsed / micro_batches:.2f}s per micro-batch; "
          f"diag said {bwd:.2f}s)")
    if not args.smoke:
        if best_val >= 0:
            # Report the SAVED checkpoint, not the last step's weights — with
            # early stopping those are usually different.
            model.load_adapter(str(ADAPTER_DIR), adapter_name="best")
            model.set_adapter("best")
            print(f"[i] using best checkpoint (val_exact_match={best_val:.1%})")
        else:
            ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ADAPTER_DIR))
            tok.save_pretrained(str(ADAPTER_DIR))
            print(f"[i] adapter saved -> {ADAPTER_DIR}")
        print("[i] test-set score:")
        print(json.dumps(evaluate(model, tok, test_rows, verbose=True,
                                  out_path=args.out), indent=2))
    else:
        print("\n[smoke OK] it fits and loss moved. Now run the full training:")
        print(f"    python train.py --base {args.base}"
              f"{' --no-quant' if args.no_quant else ''}")
        print("Compare against BASELINE.md (exact_match 66.7%, latency p50 626ms).")


if __name__ == "__main__":
    main()
