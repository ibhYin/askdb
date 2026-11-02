PY := .venv/bin/python
PORT ?= 8020

.PHONY: help setup db check eval selfcheck test compare

help:
	@echo "setup      创建虚拟环境并安装依赖"
	@echo "db         生成样例数据库"
	@echo "check      体检评测集（gold SQL 可执行且非空）"
	@echo "selfcheck  自检评测器（必须 100%）"
	@echo "eval       跑一轮评测"
	@echo "test       跑单元测试"
	@echo "web        启动问数界面"
	@echo "bird-data  下载 BIRD 题目与数据库"
	@echo "bird-eval  跑 BIRD 评测"
	@echo "retrieval  离线测 schema 检索的表召回率"
	@echo "mistakes   查看错题明细"

setup:
	python3.13 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

db:
	$(PY) scripts/build_sample_db.py

check:
	$(PY) scripts/check_questions.py

selfcheck:
	$(PY) -m askdb.eval --self-check

eval:
	$(PY) -m askdb.eval

test:
	$(PY) -m pytest

compare:
	@echo "用法：make compare A=runs/a.json B=runs/b.json"
	$(PY) -m askdb.eval.compare $(A) $(B)

web:
	$(PY) -m uvicorn askdb.web:app --reload --port $(PORT)

bird-data:
	$(PY) scripts/fetch_bird.py

bird-check:
	$(PY) scripts/check_bird.py

bird-eval:
	$(PY) -m askdb.eval --bird

retrieval:
	$(PY) scripts/eval_retrieval.py --top-k 6

mistakes:
	$(PY) scripts/show_mistakes.py --limit 10
