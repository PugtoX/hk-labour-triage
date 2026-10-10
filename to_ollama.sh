#!/usr/bin/env bash
# Register the fine-tuned model with Ollama.
#
# Ollama's safetensors importer is MLX-only (Apple Silicon), so pointing
# `ollama create` at a plain HF checkpoint fails on Linux/WSL with
# "MLX runtime is not available". The supported path is llama.cpp's converter
# -> GGUF -> ollama create.
#
#   bash to_ollama.sh
#
# Prerequisites: export_model.py has already produced models/triage-merged/.
set -euo pipefail

WSL=~/projects/hk-labour-triage
LLAMA=~/llama.cpp
MERGED="$WSL/models/triage-merged"
GGUF="$MERGED/triage-1.7b-f16.gguf"
NAME=triage-1.7b

cd "$WSL"

if [ ! -f "$MERGED/config.json" ]; then
    echo "[!] $MERGED not found — run 'python export_model.py' first"
    exit 1
fi

if [ ! -d "$LLAMA" ]; then
    echo "[i] cloning llama.cpp (one time) ..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA"
fi

# convert_hf_to_gguf.py needs the `gguf` package. llama.cpp ships gguf-py, so use
# that copy rather than installing anything — this venv was built with uv and has
# no pip, which is why a bare `pip install` fell through to the system Python
# 3.14 and died on PEP 668 (externally-managed-environment).
if ! python -c "import gguf" 2>/dev/null; then
    if [ -d "$LLAMA/gguf-py" ]; then
        export PYTHONPATH="$LLAMA/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
        echo "[i] using llama.cpp's bundled gguf-py"
    else
        echo "[!] no gguf module and no gguf-py in llama.cpp"
        exit 1
    fi
fi
python -c "import gguf; print('[i] gguf from', gguf.__file__)"

echo "[i] converting to GGUF (f16, ~3.4GB) — this takes a minute ..."
python "$LLAMA/convert_hf_to_gguf.py" "$MERGED" --outfile "$GGUF" --outtype f16
ls -lh "$GGUF"

# Reuse the Modelfile export_model.py already wrote and swap only the FROM line,
# so the SYSTEM prompt keeps coming from schema.py rather than a second
# hand-copied version that can drift.
sed 's|^FROM .*|FROM ./triage-1.7b-f16.gguf|' "$MERGED/Modelfile" > "$MERGED/Modelfile.gguf"
echo "[i] Modelfile.gguf:"
head -2 "$MERGED/Modelfile.gguf"

echo "[i] registering with Ollama ..."
ollama create "$NAME" -f "$MERGED/Modelfile.gguf"
ollama list

echo
echo "[next] bash /mnt/e/AI/AI_Agent/workplace/hk-labour-triage/run.sh \\"
echo "         \"python evaluate.py --model $NAME --data data/test.jsonl --out v4_ollama_ft.json\""
