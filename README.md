# askdb — 自然语言问数 Agent（评测驱动）

把中文问题变成 SQL、在真实数据库上执行、从报错里自己改过来。
重点不是"能生成 SQL"，而是**每一点提升都有分数作证**。

## 现在能做什么

```bash
make web                          # 打开 http://127.0.0.1:8020 直接提问
python -m askdb.eval              # 跑一轮评测，报告存 runs/*.json
python -m askdb.eval --self-check # 自检：gold SQL 冒充模型，必须 100%
python -m askdb.eval --limit 5    # 只跑前 5 题
python -m askdb.eval.compare runs/A.json runs/B.json   # 两次评测的差异
```

没有 API Key 也能跑——会自动使用 mock 模型，用于验证流水线本身。

## Web 界面

`make web` 之后打开 `http://127.0.0.1:8020`（端口被占就 `make web PORT=8030`）：输入中文问题，SQL、执行结果表、
每轮耗时与 token 都会逐步推送到页面上。用的是 SSE，不是轮询。

界面刻意做成"trace 优先"而不是"聊天优先"——这个项目要展示的是过程，
不是让模型显得会说话。

## BIRD 公开基准

```bash
python scripts/fetch_bird.py --list    # 看题目涉及哪 11 个库
python scripts/fetch_bird.py           # 下载题目 + 数据库（约 9GB，可断点续传）
python -m askdb.eval --bird --limit 100            # 前 100 题（集中于同几个库）
python -m askdb.eval --bird --sample 150           # 等距抽样，跨 11 个库
python -m askdb.eval --bird --concurrency 3        # 并发跑分，结果顺序不变
python -m askdb.eval --bird --schema-retrieval     # 只把相关表放进 prompt
python -m askdb.eval --bird --prompt v2            # 换成"先想后写"的提示词
```

### 已测过的消融（150 题抽样）

| 配置 | EX | 平均 token |
|---|---|---|
| 基线 | 62.7% | 3062 |
| + schema 检索 | **66.7%** | 3197 |
| + 提示词 v2 | 64.0% | 4385 |

检索是主力，提示词的性价比不高。详见 `docs/results.md` 与 `docs/decisions.md`。

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

两套：

| 数据集 | 内容 | 用途 |
|---|---|---|
| `data/sample/questions.jsonl` | 24 条中文问题（自建电商库） | 跑通链路、中文演示；题目已饱和，不适合证明提升 |
| BIRD mini_dev | 500 条英文问题，11 个真实业务库 | 公开基线，可与官方 leaderboard 对比 |

官方 baseline（SQLite EX）：DeepSeek-V4.1-Flash 61.0%、GPT-6.1 Sol 88.0%——
本项目用的正是前者，所以分数有直接参照。

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
├── web.py         # FastAPI + SSE 接口
├── web_static/    # 单文件前端
└── eval/
    ├── dataset.py # 评测集加载
    ├── failure.py # 六类失败归因
    ├── runner.py  # 跑分、指标、报告落盘
    ├── compare.py # 两次评测对比
    └── bird.py    # BIRD 多库评测集加载
scripts/
├── build_sample_db.py  # 可复现的样例电商库
└── check_questions.py  # 评测集体检
tests/                  # 34 个用例
```

## 路线图

- [x] 第 1 周：只读执行器、Agent 循环、评测骨架、24 题样例集、对比报表
- [x] 第 2 周：BIRD 接入（题目 + 11 个库）、多库评测、跑出公开基线
- [ ] 第 3 周：schema 检索（表/列召回）
- [ ] 第 4 周：执行反馈自纠 + 结果合理性校验
- [ ] 第 5 周：上下文裁剪、成本与延迟优化
- [x] 第 6 周：FastAPI + SSE 问数界面（提前完成）
- [ ] 待办：Docker 部署、上下文裁剪与成本优化
