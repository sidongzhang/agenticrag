# AgenticRAG + RL 项目 · 学习计划（带教版）

> 目标：从零完整学会这个项目的思想与技术，学完能自己复现、能改、能在面试讲透。
> 教法：每课四步 —— 一句话目标 → 贴真实代码 → 口语拆解 → 一道自测题（答对再进下一课）。
> 每课都以本仓库真实文件为锚，代码在 `/Users/zhangsidong/Desktop/core`。

---

## 一、学完你会得到什么

**能力**
1. 看懂并复现完整链路：语料 → 索引 → 多智能体 → 数据合成 → SFT → Agentic RL → 评测
2. 会搭 RAG：BGE-M3/FAISS、BM25 中英混合分词、知识图谱、RRF 融合、重排
3. 会用 LangGraph 写可循环/可分支的 Agent
4. 会造多跳训练数据：种子 QA、逐跳扩展、四重验证、质量漏斗
5. 会做 LoRA SFT，理解「SFT 格式必须与 RL 对齐」
6. 会配 veRL 多轮 rollout，会设计并调奖励
7. 会做评测：EM/F1 + LLM Judge 三维 + hop-aware + 消融

**资产**
- 一个能进简历、扛得住 42 道深挖题的项目
- 一套可复现/可开源的仓库（含消融与曲线）
- 一句话核心贡献：两阶段课程学习打破 Correctness / Faithfulness 跷跷板

**前提**：资料公开、面试官见得多 → 必须做「自己的增量」（换垂域语料 or 加 DAPO/GSPO 消融），才算你的项目。

---

## 二、简历目标稿（Agent RL 方向）

**面向中文金融多跳问答的 AgenticRAG 与多轮工具调用 GRPO** · 独立完成

技术栈：Python / PyTorch / Qwen3-4B / LLaMA-Factory / veRL / LangGraph / FAISS+BGE-M3 / BM25 / BGE-Reranker / vLLM / GRPO / ReAct

项目介绍：从零构建金融垂域多跳检索问答系统，覆盖语料处理 → 数据合成 → 多智能体架构 → SFT → Agentic GRPO 全链路；核心贡献是提出两阶段课程学习策略，打破 Correctness 与 Faithfulness 的跷跷板，是十余个 GRPO 版本中唯一同时在两个维度超越 SFT 基线的方案。

核心工作：
1. 垂域语料与三路检索：7 家零售上市公司半年报 → 1,933 chunks；BGE-M3(FAISS) + BM25(中英混合分词) + 知识图谱(LLM 抽三元组/万级节点) 三路 RRF 融合 + CrossEncoder 重排。
2. 数据合成 pipeline 与质控：种子 QA → 逐跳扩展(inference/comparison) → 四重验证 → 规则清洗 + LLM Judge(阈值 13/15) → 可达性过滤；4,000 种子 → 797 训练 + 185 测试。
3. PEV 多智能体系统：LangGraph 上实现 Planner/Executor/Verifier/Synthesizer 循环，9 轮迭代；定位出 replan 质量是根因，改为把「已搜 tool+query」反馈给 Planner 后首次超越 baseline。
4. Agentic GRPO：veRL 多轮 tool_agent（4×A100，最多 7 轮，hermes 格式）；复合奖励 hop_pr×0.30 + Faith×0.25 + Corr×0.25 + grounded×0.10 + format×0.10；两阶段课程使 Judge_C 0.279→0.334、Faith 0.169→0.199，同时超越 SFT base。
5. 评测体系：EM/F1 + LLM Judge 三维 + hop-aware 诊断 + 工具/组件/规模三组消融。

---

## 三、反推学习表（简历 → 课程）

| 简历里那句话 | 背后要掌握的知识 | 课程 |
|---|---|---|
| 7 家公司半年报 → 1,933 chunks | PDF 解析、chunk 切分与重叠 | 第 1 课 |
| BGE-M3 + FAISS / BM25 / 知识图谱 | 向量检索、BM25、中英混合分词、三元组抽取 | 第 1、2 课 |
| RRF 融合 + CrossEncoder 重排 | 多路融合、重排原理 | 第 2 课 |
| 种子 QA → 逐跳扩展 | 合成数据原理、inference vs comparison | 第 7 课 |
| 四重验证 + 规则清洗 + LLM Judge + 可达性 | 数据质控、防幻觉/作弊/不可达 | 第 8 课 |
| PEV 多智能体 + 9 轮迭代 | LangGraph、状态机、replan 质量 | 第 3–6 课 |
| SFT / LoRA / ReAct | PEFT、对话模板、loss mask | 第 9、10 课 |
| Agentic GRPO（多轮 tool_agent） | 为什么端到端、veRL 多轮配置 | 第 11、12 课 |
| 复合奖励 + 两阶段课程 | 奖励设计、信号密度、curriculum | 第 13、14 课 |
| 评测 + 消融 | EM/F1、LLM Judge、hop-aware、消融 | 第 15 课 |
| 面试能讲透 | 全链路复述 + 42 道深挖题 | 第 16 课 |

---

## 四、路线（7 阶段 16 课）

### 阶段一 · 建立地图
- [ ] **第 0 课｜项目全景**：代码在干嘛、模块怎么串、数据两条 pipeline

### 阶段二 · 检索的眼睛
- [ ] **第 1 课｜索引怎么建**：build_index.py → FAISS / BM25 / chunk 元数据
- [ ] **第 2 课｜四路检索 + RRF 融合 + 重排**：为什么 v9 靠它拿最优 F1

### 阶段三 · Agent 骨架
- [ ] **第 3 课｜共享状态与流程图**：agents/state.py + graph.py
- [ ] **第 4 课｜Router + Planner**：单跳/多跳分类、子查询与工具选择
- [ ] **第 5 课｜Executor + Verifier**：工具调用、证据充分性、replan
- [ ] **第 6 课｜Synthesizer + Prompt 分级 + v1→v9 优化史**

### 阶段四 · 数据合成
- [ ] **第 7 课｜种子 QA → 逐跳扩展**（inference vs comparison）
- [ ] **第 8 课｜四重验证 + 清洗 + LLM Judge + 训测划分**

### 阶段五 · SFT
- [ ] **第 9 课｜Oracle 轨迹 → ReAct/messages 训练数据**
- [ ] **第 10 课｜LoRA SFT + 格式对齐四返工**

### 阶段六 · Agentic RL（核心）
- [ ] **第 11 课｜为什么端到端训 Search Agent，而非只训 Planner**
- [ ] **第 12 课｜veRL 多轮 rollout 配置（tool_agent/hermes/HTTP 检索/显存）**
- [ ] **第 13 课｜奖励 v9a 逐项拆解**
- [ ] **第 14 课｜Reward 迭代史 + 两阶段课程学习**

### 阶段七 · 评测与收口
- [ ] **第 15 课｜评测方法论：EM/F1 + LLM Judge + hop-aware + 消融**
- [ ] **第 16 课｜全链路串讲 + 42 道深挖题演练**

---

## 五、关键数字备忘（简历/面试用）

- 语料：7 家零售上市公司 2025 半年报 → 1,933 chunks（500 字符/块，50 重叠，PyMuPDF）
- 检索：BGE-M3(1024 维, L2 归一化, 内积) + BM25(中英混合分词) + 知识图谱(万级节点)
- 消融：关键词贡献最大（去掉 F1 −0.018）；图谱在 3–4 跳复杂比较类 +0.075~+0.098 F1；v9 RRF F1=0.575 最优
- 数据漏斗：4,000 种子 → 2,212 原始 → 1,708 规则清洗 → 1,340 LLM Judge(均分 14.4/15) → 797 训练 + 185 测试
- 4hop：1,000 种子只生成 64 条 → 改策略后 13→107 条
- SFT base：Judge_C 0.279 / Faith 0.169 / CtxP 0.288 / F1 0.281 / EM 0.178
- 当前最优 v14e s30：Judge_C 0.334 / Faith 0.199 / CtxP 0.251 / F1 0.267 / EM 0.130（唯一同时超 SFT base）
- GRPO：Qwen3-4B / 4×A100-40GB / tool_agent / hermes / max_assistant_turns=7 / n=4 / kl 0.05
- Reward v9a：hop_pr×0.30 + Faith×0.25 + Corr×0.25 + grounded×0.10 + format×0.10 − 搜索不足惩罚
- 关键发现：两阶段课程打破跷跷板；硬门控(v12e)完全失败；val reward 峰值 ≠ 评测最优
