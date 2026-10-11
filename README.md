# AgenticRAG + Agentic GRPO｜中文金融多跳问答

面向**中文金融领域多跳问答**的检索增强 Agent 与强化学习后训练系统。覆盖从
**语料处理 → 数据合成 → 多智能体检索 → SFT → Agentic GRPO → 评测**的完整链路，
训练小模型（Qwen3-4B）自主完成多轮检索与推理。

> 核心贡献：提出**两阶段课程学习（先 grounding、后 correctness）**，打破
> Correctness 与 Faithfulness 之间的跷跷板，是十余个 GRPO 版本中**唯一同时在两个
> 维度超越 SFT 基线**的方案。

---

## 一、结果速览

**主指标（185 条中文金融测试集，LLM Judge 三维 + EM/F1）**

| 模型 | Judge_C | Faith | CtxP |
|---|---:|---:|---:|
| SFT base | 0.279 | 0.169 | 0.288 |
| 单阶段 GRPO（v9e） | 0.324 | 0.102 | 0.200 |
| 单阶段 GRPO（v11e，grounding-first） | 0.196 | 0.225 | 0.209 |
| **两阶段 GRPO（v14e s30）** | **0.334** | **0.199** | **0.251** |

- 单阶段训练中 **Correctness 与 Faithfulness 是跷跷板**：调高一个、另一个必掉。
- 两阶段训练用「先学基于证据回答、再学答对」的顺序，**首次让两个维度同时超过 SFT base**。
- 关键证据：v14e 与 v9e 使用**完全相同的 reward**，仅 base 不同，Faith 即高出 +0.097，
  说明收益来自**训练顺序**（curriculum learning），而非更好的 reward。

---

## 二、系统架构

```
语料(PDF) ──切块──> chunks(1,933) ──> 三路检索索引
                                        ├─ FAISS + BGE-M3（语义）
                                        ├─ BM25（中英混合分词）
                                        └─ 知识图谱（LLM 抽实体关系）
用户问题 ──Router──┬─ 简单 → 单次 RAG 回答
                   └─ 多跳 → PEV 循环：Planner→Executor→Verifier（不足则 replan）→ Synthesizer
训练：veRL 多轮 tool_agent rollout（hermes 协议）→ 复合规则奖励 → GRPO
```

### 模块

| 目录 | 作用 |
|---|---|
| `retrieval/` | 四路检索工具：语义 / 关键词 / 图谱 / RRF 混合融合 + BGE-Reranker 重排 |
| `agents/` | LangGraph 多智能体：Router / Planner / Executor / Verifier / Synthesizer |
| `scripts/` | 数据合成 pipeline：种子 QA → 逐跳扩展 → 四重验证 → 清洗 → 训测划分 |
| `training/` | SFT（LLaMA-Factory）与 Agentic GRPO（veRL）脚本、奖励函数 |
| `evaluation/` | EM/F1、LLM Judge 三维、hop-aware 诊断、消融实验 |
| `llm/` | LLM 统一调用（本地 vLLM + 云端 API） |

---

## 三、关键设计

- **三路检索 + RRF 融合**：语义/关键词/图谱互补，RRF 按排名融合（不比分数比名次），
  再用 CrossEncoder 精排；消融显示关键词贡献最大，图谱在 3–4 跳复杂比较类上增益明显。
- **PEV 多智能体**：v1→v9 迭代定位出 **replan 质量**是瓶颈——把「已搜过的工具 + 查询」
  回传给 Planner 后首次超越基线（限制迭代次数、证据去重等方案均失败）。
- **数据合成**：用**反证**过滤假多跳（不检索可答、仅单文档可答均丢弃），并做**可达性过滤**
  （每跳 gold chunk 必须能被检索命中）。
- **Agentic GRPO**：多轮 tool-agent（max 7 turns、n=4、hermes 格式），复合奖励
  `hop_pr×0.30 + Faith×0.25 + Corr×0.25 + grounded×0.10 + format×0.10 − 搜索不足惩罚`；
  对零方差组做自适应重采样，并修复 bf16 浮点残差导致的**伪优势**数值 bug。
- **评测**：EM/F1 + LLM Judge 三维 + hop-aware 诊断（过早终止 / 过度检索）+ 三组消融。

---

## 四、环境准备

```bash
# 环境 1：推理 / 评测 / 数据合成 / SFT
conda create -n agenticrag python=3.11
conda activate agenticrag
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# SFT 额外：cd LLaMA-Factory && pip install -e .

# 环境 2：GRPO 训练（与上面隔离）
conda create -n verl python=3.12
conda activate verl
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
cd verl && pip install -e . && cd ..
pip install -r requirements-verl.txt
```

模型：`BAAI/bge-m3`、`BAAI/bge-reranker-v2-m3` 下载到 `models/`。

---

## 五、复现流程

```bash
# 1) 构建索引（FAISS + BM25 + 知识图谱）
python scripts/build_index.py
python scripts/build_knowledge_graph.py

# 2) 数据合成
python scripts/gen_seed_qa.py
python scripts/domain_multihop_synthesis.py --lang zh
python scripts/judge_synthesis.py && python scripts/clean_synthesis.py
python scripts/split_train_test.py && python scripts/annotate_search_tools.py

# 3) SFT（Oracle 轨迹 → ReAct/messages → LoRA）
python scripts/build_oracle_traces.py
python scripts/trace_to_sft.py --lang zh
bash training/sft_pipeline.sh

# 4) Agentic GRPO（需先起检索服务）
bash training/start_retrieval_server.sh
export VERL_DIR=$(pwd)/verl
bash training/start_grpo.sh

# 5) 评测
python scripts/eval_agentic.py --model <model> --max-turns 7 --max-samples 185
```

---

## 六、本仓库说明（源码版）

为保证仓库轻量，**以下内容未纳入版本管理**（见 `.gitignore`），需按上文自行准备：

- `data/`（语料、索引、合成 QA、SFT/GRPO 数据）——由 pipeline 生成；
- `verl/`、`LLaMA-Factory/`——第三方训练框架，按 README 自行 clone；
- `models/`、`outputs/`、`results/`、训练产物与日志。

仓库仅包含**项目自身源码**（`agents/ retrieval/ evaluation/ training/ scripts/ llm/ api/` 等）。

---

## 七、致谢与参考

- 评测基准：**AgenticRAGTracer**（本项目在其基础上完成复现与端到端改造）。
- 训练框架：**veRL**（GRPO 多轮 rollout）、**LLaMA-Factory**（SFT）。
- 模型与检索：**Qwen3-4B**（基座）、**BGE-M3 / BGE-Reranker-v2-m3**（检索与重排）。
- 数据/许可按各自来源保留，仓库仅含项目源码。

