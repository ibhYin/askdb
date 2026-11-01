"""拉取 BIRD mini_dev 的 SQLite 数据库（不需要安装 Docker）。

BIRD 官方把数据库放在 Docker 镜像里（每个库一个 tag），这里直接用 Registry API
下载对应平台的 layer，只解出 .sqlite 文件。这样不用装 Docker、也不用下整个镜像。

用法：
    python scripts/fetch_bird.py --list
    python scripts/fetch_bird.py --db toxicology
    python scripts/fetch_bird.py            # 全部 11 个库
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REGISTRY = "https://registry-1.docker.io"
REPO = "3036197201/bird-mini-dev-sqlite"
JSONL_URL = "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_sqlite.jsonl"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIRD_DIR = PROJECT_ROOT / "data" / "bird"
DATABASES_DIR = BIRD_DIR / "databases"
QUESTIONS_PATH = BIRD_DIR / "mini_dev_sqlite.jsonl"

ACCEPT_INDEX = ", ".join(
    [
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ]
)
ACCEPT_MANIFEST = ", ".join(
    [
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
    ]
)


def _get(url: str, token: str | None = None, accept: str | None = None):
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if accept:
        request.add_header("Accept", accept)
    return urllib.request.urlopen(request, timeout=120)


def _json(url: str, token: str | None = None, accept: str | None = None) -> dict:
    with _get(url, token, accept) as response:
        return json.load(response)


def auth_token() -> str:
    payload = _json(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull"
    )
    return payload["token"]


def list_tags(token: str) -> list[str]:
    payload = _json(f"{REGISTRY}/v2/{REPO}/tags/list", token)
    return payload.get("tags", [])


def tag_for(db_id: str, tags: list[str]) -> str | None:
    prefix = f"harbor-minidev1024-r1-{db_id}-"
    return next((tag for tag in tags if tag.startswith(prefix)), None)


def platform_manifest_digest(tag: str, token: str, arch: str = "amd64") -> str:
    index = _json(f"{REGISTRY}/v2/{REPO}/manifests/{tag}", token, ACCEPT_INDEX)
    for entry in index.get("manifests", []):
        platform = entry.get("platform", {})
        if platform.get("os") == "linux" and platform.get("architecture") == arch:
            return entry["digest"]
    raise RuntimeError(f"{tag} 没有 linux/{arch} 平台的镜像")


def download_database(db_id: str, token: str, arch: str = "amd64") -> Path:
    """下载一个库的所有 layer，把 .sqlite 解到 data/bird/databases/<db_id>/。"""
    tags = list_tags(token)
    tag = tag_for(db_id, tags)
    if tag is None:
        raise SystemExit(f"镜像里找不到 {db_id}，可用 tag：{tags}")

    digest = platform_manifest_digest(tag, token, arch)
    manifest = _json(f"{REGISTRY}/v2/{REPO}/manifests/{digest}", token, ACCEPT_MANIFEST)

    dest_dir = DATABASES_DIR / db_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for layer in manifest.get("layers", []):
        if layer.get("mediaType", "").endswith("gzip") is False and "tar" not in layer.get("mediaType", ""):
            continue
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with _get(f"{REGISTRY}/v2/{REPO}/blobs/{layer['digest']}", token) as response:
                shutil.copyfileobj(response, tmp, length=1024 * 1024)
        try:
            with tarfile.open(tmp_path, "r:gz") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".sqlite"):
                        continue
                    target = dest_dir / Path(member.name).name
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    target.write_bytes(source.read())
                    extracted += 1
        finally:
            tmp_path.unlink(missing_ok=True)

    if extracted == 0:
        raise RuntimeError(f"{db_id} 的镜像里没有 .sqlite 文件")
    return dest_dir


def fetch_questions(force: bool = False) -> Path:
    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if QUESTIONS_PATH.exists() and not force:
        return QUESTIONS_PATH
    print(f"下载题目：{JSONL_URL}")
    with _get(JSONL_URL) as response, QUESTIONS_PATH.open("wb") as target:
        shutil.copyfileobj(response, target)
    return QUESTIONS_PATH


def db_ids_from_questions(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            db_id = json.loads(line)["db_id"]
            if db_id not in ids:
                ids.append(db_id)
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 BIRD mini_dev 的 SQLite 数据库")
    parser.add_argument("--db", default=None, help="只下载指定 db_id")
    parser.add_argument("--list", action="store_true", help="只列出题目涉及的库")
    parser.add_argument("--arch", default="amd64", help="镜像架构，默认 amd64")
    parser.add_argument("--force-questions", action="store_true", help="重新下载题目 jsonl")
    args = parser.parse_args(argv)

    questions = fetch_questions(force=args.force_questions)
    db_ids = db_ids_from_questions(questions)

    if args.list:
        print(f"{len(db_ids)} 个库：")
        for db_id in db_ids:
            print(f"  {db_id}")
        return 0

    targets = [args.db] if args.db else db_ids
    token = auth_token()
    for index, db_id in enumerate(targets, 1):
        existing = DATABASES_DIR / db_id
        if existing.exists() and any(existing.glob("*.sqlite")):
            print(f"[{index}/{len(targets)}] {db_id} 已存在，跳过")
            continue
        print(f"[{index}/{len(targets)}] 下载 {db_id} …", flush=True)
        path = download_database(db_id, token, args.arch)
        size_mb = sum(f.stat().st_size for f in path.glob("*.sqlite")) / 1e6
        print(f"    -> {path} ({size_mb:.1f} MB)")

    print("\n完成。数据库目录：", DATABASES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
