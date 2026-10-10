# 香港勞工諮詢分流 · LoRA 微調

把市民口語化的香港勞工法諮詢問題,分到 6 個類別,並判斷緊急程度與是否需要轉人工。

**目標不是「比大模型更準」,而是「用更低的成本做到同級效果」—— 並用統計檢驗證明它做到了。**

```json
輸入: "老闆拖咗我三個月工資，我可以點算？"
輸出: {"category": "工資與扣薪", "urgency": "high", "escalate": true}
```

---

## 結果

同一份測試集(106 條,零重複)。**A 與 C 跑在同一套推理路徑上**(Ollama + 同一份 prompt 契約),所以這一對是可比的:

| 指標 | A. `qwen3.5:9b` | **C. `Qwen3-1.7B` + LoRA** |
|---|---|---|
| `urgency_acc` | 63.2% | **63.2%** ← 打平 |
| **緊急個案召回** | 37.8% | **73.0%** |
| 緊急個案精確 | 87.5% | 73.0% |
| `low` 召回 | **80.0%** | 36.7% |
| **加權代價 /100** ↓ | 80 | **56** |

*加權代價:漏判一個緊急個案計 3 分,其餘錯誤計 1 分 —— 業務上漏掉 urgent 最貴。*

配對顯著性(McNemar,同一批題):

| 對比 | 指標 | 不一致 | p |
|---|---|---|---|
| **base → LoRA** | `exact_match` | **3 : 40** | **0.000** |
| **base → LoRA** | `urgency` | 6 : 34 | **0.000** |
| 9B vs LoRA | `exact_match` | 13 : 25 | 0.073(邊緣,偏向 C) |
| 9B vs LoRA | `urgency` | 22 : 22 | 1.000(打平) |

**三條結論:**

1. **微調極顯著有效** —— 相對同模型的 base,`exact_match` 是 3:40(p=0.000)。兩次獨立測量都確認。
2. **1.7B 換來的不是「整體更準」,而是兩件具體的事** —— **緊急個案召回翻倍**(37.8% → 73.0%),
   **加權代價低 30%**(80 → 56),外加模型小一半。整體準確率是**打平**。
3. **代價在 `low` 檔**:召回從 80.0% 掉到 36.7% —— 它把所有東西往中間檔拉。
   **值不值得,取決於業務怎麼給「漏掉緊急」和「誤報緊急」定價。**

> base(`Qwen3-1.7B` 未微調)走 HF transformers,因為 9B fp16(13.4GB)裝不進 8GB 顯卡,
> 而 1.7B 的 HF 版是驗證微調效果最乾淨的對照。**這個跨框架差異保留在 `BASELINE.md` 的已知偏差裡。**

## 為什麼這個專案值得看:結論被推翻過三次

**同一組模型,只換測量方式,結論就變了:**

| 協議 | 變化 | 當時的結論 | 後來怎麼了 |
|---|---|---|---|
| v1 | n=46(含 13 條重複) | 「9B 更好,1.7B 落後 15.2pp」 | **全部撤回**(p=0.167 不顯著) |
| v2 | 擴到 717 條(n=108) | 「1.7B 反超」 | 方向對,但數據仍含重複 |
| v3 | 修去重 bug(n=106 乾淨) | 「1.7B 反超,顯著 p=0.041」 | 成立,但依賴 HF 框架 |
| v4 | 轉 GGUF 走 Ollama | 「優勢消失(p=0.324)」 | **是腳本 bug**:SYSTEM 被加了兩次 |
| **v5** | 改用 `system` 欄位 | 「urgency 打平,緊急召回翻倍」 | **當前結論** |

**9B 在 46 條上是 73.9%,在 106 條上是 62.3% —— 波動 11.6pp。** 中間那句「瓶頸精確在 urgency」
在 n=46 時 p=0.167(**不顯著**),**已被撤回**。

**這件事本身是這個專案最想展示的能力:知道自己什麼時候不該相信自己的數字。**

---

## 流程

```
                    ┌─────────────────────────────────────────┐
                    │  合成數據 (synthesize.py)                │
                    │  本地 qwen3.5:9b 生成問題 + 標籤          │
                    └────────────────┬────────────────────────┘
                                     ▼
        人工抽檢 (review.py) ──► 標籤體系 16 類 → 6 類(30% 標錯過)
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  劃分 (check_dataset.py)                 │
                    │  去重 · 分層 70/15/15 · 主題均衡          │
                    └────────────────┬────────────────────────┘
                                     ▼
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  基線 (evaluate.py)          微調 (train.py)              分析 (analyze.py)
  qwen3.5:9b                  Qwen3-1.7B + fp16 LoRA        McNemar / CI / cost
  Ollama 路徑                  HF 訓練 → GGUF → Ollama       逐類 recall+precision
```

---

## 快速開始

```bash
# 環境:WSL2 + RTX 3060 Ti 8GB · Python 3.12
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch transformers peft bitsandbytes

python synthesize.py --per-category 120     # 合成 ~720 條
python check_dataset.py                     # 去重 + 劃分
python check_facts.py                       # 虛構機構／外地法例掃描

# 把 1.7B 也部署到 Ollama,才能和 9B 同框架比
python export_model.py                      # 合併 LoRA → models/triage-merged/
bash to_ollama.sh                           # llama.cpp → GGUF f16 → ollama create

# 三方對比
python evaluate.py --model qwen3.5:9b  --data data/test.jsonl --out v5_9b.json
python evaluate.py --model triage-1.7b --data data/test.jsonl --out v5_ft.json
python train.py --eval-only --no-adapter --no-quant --base Qwen/Qwen3-1.7B --out v3_base.json
python train.py --eval-only --no-quant --base Qwen/Qwen3-1.7B --out v3_ft.json

# 全套統計
python analyze.py v5_9b.json v5_ft.json      # 主表 + McNemar + cost
python analyze.py v3_base.json v3_ft.json    # 微調是否有效
```

沒有 GPU 也能跑 `synthesize.py` / `check_dataset.py` / `analyze.py` —— 它們只需要 Python 標準庫。

---

## 踩過的坑

| # | 坑 | 症狀 | 修法 |
|---|---|---|---|
| 1 | **標籤體系照搬業務目錄** | 16 類抽檢 30% 標錯 | 合併成 6 類,並補「不包含」條款 |
| 2 | **`format:"json"` 不約束結構** | 16 主題裡 13 個只出 1 條 | 改用 JSON Schema 強制陣列 |
| 3 | **QLoRA 在小模型上是負優化** | 4-bit 比 fp16 **慢 20 倍**、更費顯存 | 1.7B 以下用 fp16 LoRA |
| 4 | **缺控制組** | 差點認定「微調無效」 | 加 `--no-adapter` 跑同路徑 base |
| 5 | **自標自測** | 9B 的 73.9% 是「自一致率」而非準確率 | 已知限制,寫進文件 |
| 6 | **詳細規則反而有害** | 完整規則 vs 精簡規則 **p=0.003** | 保留精簡版;規則不是標籤的驅動因素 |
| 7 | **去重寫法錯誤** | 同一響應內的重複漏過,污染 1.8% | 逐條立即標記 `seen` |
| 8 | **預設值三連坑** | `MAX_LEN=512`(太短)、`--max-per-category=20`(砍掉 2/3 數據)、單次請求 69 條(JSON 截斷) | 全部改成「按實際規模」並加告警 |
| 9 | **`num_predict` 未設** | 單條推理跑 480 秒 | 限制輸出 64 token、`num_ctx` 降到 2048 |
| 10 | **SYSTEM 被加了兩次** | 自定義模型收到 Modelfile + prompt 兩份規則,緊急召回從 75.7% 掉到 51.4% | 改用 `/api/generate` 的 `system` 欄位(覆蓋 Modelfile) |
| 11 | **WSL2 的 CUDA 退化** | 同一請求從 0.5s 變成 **56s** | 與代碼無關的基準請求 + `nvidia-smi` 排除硬件 → `wsl --shutdown` |

---

## 數據質量檢查

| 檢查項 | 工具 | 結果 |
|---|---|---|
| 重複文本 | `check_dataset.py` | **0**(去重後 704 條) |
| 標籤衝突(同文本不同標籤) | `check_dataset.py` | **0** |
| **虛構機構／外地法例** | `check_facts.py`(22 條模式) | **0 處** ✅ 達 SPEC §4.1 |
| 人工抽檢 `category` | `review.py` | **36/36 = 100%** |
| 人工抽檢 `urgency` | `review.py` | **17/20 = 85%** ⚠️ 未達 90% 線 |

## 已知限制

1. **標籤質量 85%**(人工審 20 條,SPEC 要求 ≥90%)。且 gold 由 9B 自標,不是外部真值。
2. **base 跨框架**:A(9B)與 C(LoRA)已同為 Ollama 路徑,但 B(base)走 HF transformers。
   無法消除 —— 9B fp16 需 13.4GB,裝不進 8GB 顯卡。
3. **模型會向中間檔坍縮**:18 條「兩模型一致但與 gold 不同」中,9 條 high→medium、8 條 low→medium,
   雙向對稱 —— `low` 召回只有 36.7%。
4. **n=106**:`urgency` 的 95% CI 仍約 ±10pp,所以「打平」與「邊緣顯著(p=0.073)」都應謹慎解讀。

---

## 檔案

| 檔案 | 作用 |
|---|---|
| `SPEC.md` | 任務規格、標籤定義、驗收線 |
| `BASELINE.md` | **完整實驗結果 + 統計檢驗 + 協議敏感性** |
| `schema.py` | 標籤空間的單一真相來源(訓練／評測／推理共用) |
| `synthesize.py` | LLM 合成數據(分批、去重、Schema 約束) |
| `check_dataset.py` | 去重、質量報告、分層劃分、主題均衡 |
| `check_facts.py` | 虛構機構／外地法例掃描 |
| `review.py` | 人工抽檢(三欄位各自對照驗收線) |
| `train.py` | fp16 LoRA 訓練 + 早停 + 逐條落盤 |
| `export_model.py` | 合併 LoRA 成獨立模型 + 產生 Modelfile |
| `to_ollama.sh` | llama.cpp 轉 GGUF → `ollama create` |
| `evaluate.py` | Ollama 側評測(含重試、可見錯誤) |
| `analyze.py` | per-class recall/precision、Wilson CI、McNemar、加權代價 |
| `run.sh` | 執行橋:同步程式碼 → 跑 → 輸出落到可讀取的日誌 |
| `tests/` | 4 套測試(解析器、劃分、續跑安全、評測) |
