# 實驗結果

三方對比 + 配對顯著性檢驗。**這張表是本項目的核心交付物。**

---

# 主表（協議 v5 — 同框架、乾淨數據）

> **協議指紋**：`test.jsonl` **n=106，零重複** ｜ `SYSTEM` = 379 字精簡版 ｜ gold = 9B 自標
> ｜ 推理：Ollama `/api/generate` + `system` 欄位（覆蓋 Modelfile，規則只出現一次）

## 結果

| 指標 | **A. 9B**（Ollama） | **C. 1.7B LoRA**（Ollama） |
|---|---|---|
| `urgency_acc` | 63.2% (67/106) | **63.2% (67/106)** |
| **緊急個案召回** | 37.8% (14/37) | **73.0% (27/37)** |
| 緊急個案精確 | 87.5% | 73.0% |
| `low` 召回 | **80.0%** (24/30) | 36.7% (11/30) |
| **加權代價 /100** ↓ | 80 | **56** |

配對顯著性（McNemar，同一批 106 條）：

| 對比 | 指標 | 不一致 | p | 判定 |
|---|---|---|---|---|
| A vs C | `urgency` | 22 : 22 | **1.000** | ⚠️ **完全打平** |
| A vs C | `exact_match` | 13 : 25 | **0.073** | ⚠️ 邊緣，偏向 C |

## 結論：1.7B 的價值不在「整體更準」

**urgency 總分完全打平（63.2% vs 63.2%），但分佈截然相反。** 1.7B 換來的不是準確率，而是：

1. **緊急個案召回翻倍**（37.8% → 73.0%）—— 業務上最貴的一檔，漏判代價最高
2. **加權代價低 30%**（80 → 56）
3. **模型小一半**（1.7B vs 9B，3.4GB vs 6.6GB）

**代價是 `low` 檔召回掉了一半**（80.0% → 36.7%）：它把所有東西往中間檔拉。**這個取捨值不值得，取決於業務怎麼給兩類錯誤定價** —— 而 SPEC §5 目前的指標表裡沒有代價項。

---

# 為什麼有兩套主表：框架問題

**9B fp16 ≈ 13.4GB，裝不進 8GB 顯卡**，所以 9B 只能用量化版走 Ollama。而 1.7B 的 base 走 HF transformers。**三方對比因此無法完全同框架**：

| 協議 | 9B | 1.7B base | 1.7B LoRA | 框架一致性 |
|---|---|---|---|---|
| **v3** | Ollama | HF | **HF** | 9B 與其餘不同 |
| **v5** | Ollama | （沿用 v3） | **Ollama** | A↔C **同框架** ✅；base 仍跨框架 ⚠️ |

**v5 讓「9B vs LoRA」這一對變成同框架同代碼 —— 那是最關鍵的一對。** base 的跨框架差異保留為已知偏差。

## v3（HF 路徑）作為對照

| 指標 | 9B（Ollama） | 1.7B base（HF） | 1.7B LoRA（**HF**） |
|---|---|---|---|
| `urgency_acc` | 62.3% | 42.5% | **68.9%** |
| 緊急個案召回 | 37.8% | 91.9% | **75.7%** |
| `low` 召回 | 66.7% | **0.0%** | 46.7% |
| 加權代價 /100 | 81 | 63 | **48** |
| vs base `exact` | — | — | **p=0.000**（3:40） |
| vs base `urgency` | — | — | **p=0.000**（6:34） |
| vs LoRA `exact` | p=0.041（11:24） | — | — |

**兩套表的最硬共識：微調相對同模型 base 極顯著有效（p=0.000，兩次獨立測量都確認）。**

---

# 協議演變：結論被推翻過三次

| 協議 | 變化 | 當時結論 | 後來怎麼了 |
|---|---|---|---|
| v1 | n=46，**含 13 條重複** | 「9B 更好，1.7B 落後 15.2pp，瓶頸在 urgency」 | **全部撤回**：p=0.167 不顯著；小樣本排名不可靠 |
| v2 | 擴到 717 條（n=108） | 「1.7B 反超」 | 方向對，但數據仍含重複 |
| v3 | 修去重 bug（n=106 乾淨） | 「1.7B 反超且顯著（p=0.041）」 | 成立，但依賴 HF 框架 |
| v4 | 1.7B 轉 GGUF 走 Ollama | 「優勢消失（p=0.324）」 | **是腳本 bug**：SYSTEM 被 Modelfile 和 prompt 各加一次 |
| **v5** | 改用 `system` 欄位 | 「urgency 打平，緊急召回翻倍，代價低 30%」 | **當前結論** |

**9B 在 46 條上是 73.9%，在 106 條上是 62.3% —— 波動 11.6pp。**

**三次翻轉都不是模型變了，是測量方式變了。** 這比「微調提升 X%」更值得寫進作品集。

---

# 標籤質量：本項目的真實上限

| 檢查 | 結果 | 線 | 判定 |
|---|---|---|---|
| 人工審 20 條 urgency | **17/20 = 85.0%** | ≥90% | ❌ **FAIL** |
| 9B vs 自身標籤（同 prompt） | 73.9% | — | 自一致性上限 |
| 9B vs 1.7B 一致率 | 55.7%（v5） | — | 欄位噪聲大 |
| 詳細規則 vs 精簡規則（9B） | **p=0.003** | — | 見下 |
| 虛構機構／外地法例 | **0 處**（717 條 + 106 條） | 0 | ✅ **達 SPEC §4.1** |

**兩處結構性問題：**

1. **自標自測。** gold 由 9B 以 `temperature=0.9` 自標，評測時 9B 以 `temperature=0`
   判同一批題 —— 所以 9B 的分數是**自一致率**，不是對外部真值的準確率。
2. **契約漂移（本項目自造）。** 產出 gold 的 `synthesize.PROMPT` 含完整 `URGENCY_RULES`
   ＋三個例子＋`URGENCY_WARNING`；推理用的 `SYSTEM` 是被壓到 379 字的精簡版。

**Prompt A/B 的結果出乎預料**（`evaluate.py --system full`）：

| 9B | short（379字） | full（667字） |
|---|---|---|
| high 召回 | **41.7%** | **8.3%** |
| 加權代價 /100 | **57** | **98** |
| McNemar | — | **p=0.003** |

**完整規則顯著更差。** 根因是 `URGENCY_WARNING`（「只要問『我這個情況算不算』就是 medium」）
讓模型幾乎全選 medium。**所以 gold 並不忠實於規則文本** —— 同一套規則，標註時與推理時結果顯著不同。
**「改寫規則來提升 urgency」這條路因此被堵死。**

---

# 已知偏差

1. **base 跨框架**：v5 的 A↔C 已同框架，但 B（base）仍走 HF transformers。
   無法消除 —— 9B fp16 裝不進 8GB 顯卡。
2. **1.7B 的 Ollama 版是 f16 GGUF**（3.4GB），與 HF 版理論等價但已觀察到差異
   （v3 high 召回 75.7% vs v5 73.0%），可能來自 chat template 的細微差別。
3. **`n=106`**：`urgency` 的 95% CI 仍約 ±10pp，所以「打平」和「邊緣顯著」都應謹慎解讀。
4. **gold 未全面人工校驗**：只審了 20 條。18 條「兩模型一致但與 gold 不同」中，
   **9 條 `gold=high` 被判成 medium、8 條 `gold=low` 被判成 medium** —— 雙向對稱，
   說明模型不願使用極端檔位，而非單向標錯。

---

# 復現

```bash
python check_dataset.py                     # 704 條去重 + 分層劃分
python check_facts.py                       # 虛構資訊掃描

# 1.7B 部署到 Ollama（HF safetensors 不行：Ollama 的導入器是 MLX 專用）
python export_model.py                      # 合併 LoRA → models/triage-merged/
bash to_ollama.sh                           # llama.cpp → GGUF f16 → ollama create

# 三方
python evaluate.py --model qwen3.5:9b  --data data/test.jsonl --out v5_9b.json
python evaluate.py --model triage-1.7b --data data/test.jsonl --out v5_ft.json
python train.py --eval-only --no-adapter --no-quant --base Qwen/Qwen3-1.7B --out v3_base.json
python train.py --eval-only --no-quant --base Qwen/Qwen3-1.7B --out v3_ft.json

# 分析
python analyze.py v5_9b.json v5_ft.json      # 主表 + McNemar + cost
python analyze.py v3_base.json v3_ft.json    # 微調是否有效
```

`analyze.py` 一次輸出：逐類 recall／precision（含 Wilson CI）、加權代價（含 ×2/×3/×5 敏感性）、
配對 McNemar、`escalate × urgency` 交叉表、可疑標籤清單，以及**不可解析預測的警告 + 樣本原文**。

**永久保留的逐條 `gold`+`pred`**：`baseline_9b` `base_17b` `ft_17b`（v1）· `v2_*` · `v3_*` · `v4_ollama_ft` · `v5_*`。
**不要覆蓋** —— 它們是各協議下唯一的原始證據，也是上表「協議演變」能核對的原因。

---

# 附：兩次工具鏈事故（都曾差點改變結論）

| 事故 | 症狀 | 識破方式 |
|---|---|---|
| **去重寫法錯誤** | 同一響應內的重複漏過，test 含 13 條重複文本（1.8%） | `check_dataset.py` 的去重計數 |
| **SYSTEM 被加兩次** | 自定義模型 high 召回從 75.7% 掉到 51.4% | 裸 `curl` 與代碼無關仍慢 → 順藤摸瓜；`analyze.py` 的 `unparseable` 警告 |
| **WSL2 CUDA 退化** | 同一請求從 0.5s 變成 56s（110 倍） | 與代碼無關的基準請求；`nvidia-smi` 顯示 P0/無降頻 → 排除硬件 |
