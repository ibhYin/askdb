"""准备 BIRD mini_dev 数据：题目（HuggingFace）+ 数据库（官方 OSS zip）。

    python scripts/fetch_bird.py --list          # 看题目涉及哪些库
    python scripts/fetch_bird.py --questions-only
    python scripts/fetch_bird.py --db toxicology # 只解压一个库
    python scripts/fetch_bird.py                 # 全部（约 350MB，可断点续传）

数据库来自官方 dev.zip（约 350MB，11 个库）。mini_dev 的题目就落在 dev 库上——
train.zip（9GB / 69 个库）里没有这些 db_id，别下错了。
Docker 镜像就不用考虑了：单库约 800MB，含整套运行时。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

QUESTIONS_URL = (
    "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_sqlite.jsonl"
)
ARCHIVE_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIRD_DIR = PROJECT_ROOT / "data" / "bird"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DATABASES_DIR = BIRD_DIR / "databases"
QUESTIONS_PATH = BIRD_DIR / "mini_dev_sqlite.jsonl"
ARCHIVE_PATH = CACHE_DIR / "bird-dev.zip"
# 官方归档是套娃：dev.zip 里面还有一层 dev_databases.zip
INNER_ARCHIVE_PATH = CACHE_DIR / "bird-dev-databases.zip"


def _open(url: str, *, resume_from: int = 0):
    request = urllib.request.Request(url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
    return urllib.request.urlopen(request, timeout=120)


def fetch_questions(force: bool = False) -> Path:
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if QUESTIONS_PATH.exists() and not force:
        return QUESTIONS_PATH
    print(f"下载题目：{QUESTIONS_URL}")
    with _open(QUESTIONS_URL) as response, QUESTIONS_PATH.open("wb") as target:
        shutil.copyfileobj(response, target)
    return QUESTIONS_PATH


def download_archive(force: bool = False) -> Path:
    """下载 zip，支持断点续传（这个文件很大，断一次重下很痛）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if force and ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    if ARCHIVE_PATH.exists():
        print(f"发现已下载的归档：{ARCHIVE_PATH}（{ARCHIVE_PATH.stat().st_size / 1e9:.2f} GB），继续续传")
    else:
        print(f"下载数据库归档（约 350MB，可中断后重跑续传）：{ARCHIVE_URL}")

    offset = ARCHIVE_PATH.stat().st_size if ARCHIVE_PATH.exists() else 0
    try:
        response = _open(ARCHIVE_URL, resume_from=offset)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # 从文件末尾之后开始要，说明已经下完了
            print(f"归档已完整（{offset / 1e9:.2f} GB），跳过下载")
            return ARCHIVE_PATH
        raise
    if offset and getattr(response, "status", 200) != 206:
        print("服务端不支持断点续传，从头下载")
        offset = 0

    total = offset + int(response.headers.get("Content-Length") or 0)
    written = offset
    with ARCHIVE_PATH.open("ab" if offset else "wb") as target:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            written += len(chunk)
            if total and written % (50 * 1024 * 1024) < 1024 * 1024:
                print(f"  {written / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
    print(f"归档就绪：{ARCHIVE_PATH}（{ARCHIVE_PATH.stat().st_size / 1e9:.2f} GB）")
    return ARCHIVE_PATH


def db_ids_from_questions(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        db_id = json.loads(line)["db_id"]
        if db_id not in ids:
            ids.append(db_id)
    return ids


def _sqlite_members(archive: zipfile.ZipFile, db_id: str) -> list[zipfile.ZipInfo]:
    """在 zip 里找某个库的 .sqlite 文件，不假设目录结构。"""
    return [
        info
        for info in archive.infolist()
        if info.filename.endswith(".sqlite") and f"/{db_id}/" in info.filename
        and not info.filename.startswith("__MACOSX")
    ]


def open_databases_archive(archive_path: Path = ARCHIVE_PATH) -> zipfile.ZipFile:
    """官方 zip 是套娃结构，返回真正装着各库的那一层。"""
    with zipfile.ZipFile(archive_path) as outer:
        nested = [i for i in outer.infolist() if i.filename.endswith("databases.zip")]
        if not nested:
            return zipfile.ZipFile(archive_path)  # 扁平结构，直接用

        member = nested[0]
        if not (INNER_ARCHIVE_PATH.exists() and INNER_ARCHIVE_PATH.stat().st_size == member.file_size):
            print(f"  解开内层归档 {member.filename}（{member.file_size / 1e9:.2f} GB）…", flush=True)
            with outer.open(member) as source, INNER_ARCHIVE_PATH.open("wb") as out:
                shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
    return zipfile.ZipFile(INNER_ARCHIVE_PATH)


def extract_databases(db_ids: list[str], archive_path: Path = ARCHIVE_PATH) -> list[str]:
    missing: list[str] = []
    with open_databases_archive(archive_path) as archive:
        for db_id in db_ids:
            members = _sqlite_members(archive, db_id)
            if not members:
                missing.append(db_id)
                continue

            dest_dir = DATABASES_DIR / db_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            for member in members:
                target = dest_dir / Path(member.filename).name
                print(f"  解压 {member.filename} → {target.name} …", flush=True)
                with archive.open(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out, length=4 * 1024 * 1024)
    return missing


def verify(db_id: str) -> str:
    path = DATABASES_DIR / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(path)
    try:
        tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
    finally:
        conn.close()
    return f"{db_id}: {tables} 张表，{path.stat().st_size / 1e6:.1f} MB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="准备 BIRD mini_dev 题目与数据库")
    parser.add_argument("--db", default=None, help="只解压指定 db_id")
    parser.add_argument("--list", action="store_true", help="只列出题目涉及的库")
    parser.add_argument("--questions-only", action="store_true", help="只下载题目，不碰归档")
    parser.add_argument("--delete-archive", action="store_true", help="解压完成后删除 zip")
    parser.add_argument("--force-questions", action="store_true", help="重新下载题目")
    args = parser.parse_args(argv)

    questions = fetch_questions(force=args.force_questions)
    db_ids = db_ids_from_questions(questions)

    if args.list:
        print(f"{len(db_ids)} 个库：")
        for db_id in db_ids:
            print(f"  {db_id}")
        return 0
    if args.questions_only:
        print(f"题目已就绪：{questions}（{len(db_ids)} 个库）")
        return 0

    archive = download_archive()
    targets = [args.db] if args.db else db_ids
    missing = extract_databases(targets, archive)

    print()
    for db_id in targets:
        if db_id in missing:
            continue
        print("  " + verify(db_id))
    if missing:
        print(f"[警告] 归档里没找到这些库：{', '.join(missing)}", file=sys.stderr)

    if args.delete_archive:
        archive.unlink(missing_ok=True)
        print(f"\n已删除归档 {archive}")
    else:
        print(f"\n归档保留在 {archive}（加 --delete-archive 可删）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
