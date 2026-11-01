# askdb — 自然语言问数 Agent（评测驱动）

把中文问题变成 SQL、在真实数据库上执行、从报错里自己改过来。
重点不是"能生成 SQL"，而是**每一点提升都有分数作证**。

## 现在能做什么

```bash
python -m askdb.eval              # 跑一轮评测，报告存 runs/*.json
python -m askdb.eval --self-check # 自检：gold SQL 冒充模型，必须 100%
python -m askdb.eval --limit 5    # 只跑前 5 题
python -m askdb.eval.compare runs/A.json runs/B.json   # 两次评测的差异
```

没有 API Key 也能跑——会自动使用 mock 模型，用于验证流水线本身。

## 设计取舍

| 决定 | 原因 |
|---|---|
| 手写循环，不用 LangChain | 只有"生成→执行→回填"三步，200 行比框架更好控制上下文与重试 |
| SQL 执行严格只读 | `mode=ro` + authorizer 白名单，模型写不坏评测库 |
| 超时 + 行数上限 | 一条笛卡尔积能把评测卡死，也能把上下文撑爆 |
| 空结果也回灌 | 0 行通常意味着过滤条件写错，比报错更隐蔽 |
| 评测器先自检 | 尺子不准时，任何分数都是噪音 |
| 报告永不覆盖 | 覆盖一次就丢一次基线，比较就失去意义 |

## 指标

主指标是 **Execution Accuracy**：预测 SQL 与 gold SQL 在库上执行，结果集一致即算对
（默认无序多重集比较，浮点 4 位小数归一；`LIMIT` 类题目标记 `ordered`，顺序也参与判定）。
同时记录：平均步数、token 消耗、P50/P95 延迟、六类失败的分布。

## 评测集

当前 `data/sample/questions.jsonl`：24 条中文问题，覆盖 count / join / group_by /
having / subquery / ratio / date_func，难度分三档。第 2 周接入 BIRD 100 题作为公开基线。

题目新增后先跑体检，坏题比没题更糟：

```bash
python scripts/check_questions.py   # gold SQL 必须可执行且非空
python -m askdb.eval --self-check   # 自检必须 100%
```

## 快速开始

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 填 API_KEY 即走真实模型；留空走 mock
python scripts/build_sample_db.py
python -m askdb.eval --self-check
python -m askdb.eval
```

## 目录

```
askdb/
├── agent.py       # 循环：生成 SQL → 执行 → 错误/空结果回填 → 重试
├── db.py          # 只读执行、schema 抽取、结果比较
├── llm.py         # OpenAI 兼容客户端 + 离线 mock
├── config.py      # 配置（无 key 显式回落 mock）
└── eval/
    ├── dataset.py # 评测集加载
    ├── failure.py # 六类失败归因
    ├── runner.py  # 跑分、指标、报告落盘
    └── compare.py # 两次评测对比
scripts/
├── build_sample_db.py  # 可复现的样例电商库
└── check_questions.py  # 评测集体检
tests/                  # 34 个用例
```

## 路线图

- [x] 第 1 周：只读执行器、Agent 循环、评测骨架、24 题样例集、对比报表
- [ ] 第 2 周：BIRD 100 题，跑出第一个真实基线
- [ ] 第 3 周：schema 检索（表/列召回）
- [ ] 第 4 周：执行反馈自纠 + 结果合理性校验
- [ ] 第 5 周：上下文裁剪、成本与延迟优化
- [ ] 第 6 周：FastAPI + SSE 问数界面、Docker 部署
