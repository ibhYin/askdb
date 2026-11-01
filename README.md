# askdb — 自然语言问数 Agent（评测驱动）

把中文问题变成 SQL、在真实数据库上执行、从报错里自己改过来。
重点不是"能生成 SQL"，而是**每一点提升都有分数作证**。

## 为什么做这个

大多数 LLM 项目止步于 demo：能跑通，但说不清好坏。这个项目的全部结构
围绕一件事——**测量**：

- 100 条带 gold SQL 的评测题，执行准确率（Execution Accuracy）是主指标
- 每次改动跑一次分，结果落 JSON，可对比
- 失败自动归因到具体类型（选错表 / 连接错 / 聚合错 / 结果为空 / 运行时错误）
- 同时记录 token 成本与 P50/P95 延迟

## 路线图

| 周 | 内容 | 状态 |
|---|---|---|
| 1 | SQLite 执行器 + Agent 循环 + 评测骨架 + 基线分数 | 进行中 |
| 2 | 接入 BIRD 100 题，评测报告自动化 | |
| 3 | Schema 检索（表/列召回） | |
| 4 | 执行反馈自纠 + 结果合理性校验 | |
| 5 | 上下文裁剪 + 成本/延迟优化 | |
| 6 | FastAPI + SSE 问数界面 + 部署 | |

## 快速开始

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 可选：填 API_KEY；不填走 mock
python -m scripts.build_sample_db
python -m askdb.eval.runner --limit 20
```

## 设计取舍

- **不用 LangChain**：这里只需要"生成→执行→回填"三步循环，手写 200 行比
  引入框架更清楚，也更好控制上下文与重试策略。
- **SQL 执行严格只读**：拒绝 `INSERT/UPDATE/DELETE/DDL`，带超时与行数上限，
  防止模型把评测库写坏。
- **mock provider**：没有 API Key 也能跑通全链路与单测，CI 不需要密钥。

## 目录

```
askdb/
├── agent.py      # 循环：生成 SQL → 执行 → 错误回填 → 重试
├── db.py         # 只读执行 + schema 抽取 + 结果比较
├── llm.py        # OpenAI 兼容客户端 + mock
├── config.py     # 配置读取
└── eval/         # 数据集、跑分、失败归因
```
