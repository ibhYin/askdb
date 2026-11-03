"""测试夹具：保证样例库存在。

样例库是可再生产物（`.gitignore` 排除了 `data/**/*.db`），所以新克隆的仓库里没有它。
用夹具按需生成，比在 CI 里依赖步骤顺序更可靠——直接 `pytest` 就能跑。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DB = ROOT / "data" / "sample" / "shop.db"


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_database() -> Path:
    if not SAMPLE_DB.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_sample_db.py")],
            check=True,
            cwd=ROOT,
            capture_output=True,
        )
    return SAMPLE_DB
