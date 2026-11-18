"""Download public foundation weights and verify local reproduction assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LOCK = Path(__file__).with_name("assets.lock.json")


def asset_group(name: str) -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))[name]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_assets(name: str, directory: Path, *, hash_large: bool = True) -> int:
    files = asset_group(name)["files"]
    issues = []
    for relative, expected in files.items():
        path = directory / relative
        if not path.is_file():
            issues.append(f"缺少 {relative}")
        elif path.stat().st_size != expected["bytes"]:
            issues.append(f"大小不符 {relative}")
        elif (hash_large or expected["bytes"] < 64 * 2**20) and sha256(path) != expected["sha256"]:
            issues.append(f"SHA256 不符 {relative}")
    if issues:
        raise ValueError(f"{name} 资产校验失败（{directory}）\n" + "\n".join(issues))
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "check"))
    parser.add_argument("group", choices=("embedding_2b", "teacher_27b", "learned", "adapter", "tongagents"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    entry = asset_group(args.group)
    if args.action == "download":
        if "repo_id" not in entry:
            parser.error("训练资产和 TongAgents 安装包由授权渠道提供，仅支持 check。")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            parser.error("下载底座需要 pip install huggingface-hub==1.7.2。")
        snapshot_download(
            repo_id=entry["repo_id"],
            revision=entry["revision"],
            local_dir=directory,
            allow_patterns=list(entry["files"]),
        )
    try:
        count = verify_assets(args.group, directory)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"{args.group}: {count} 个文件的大小和 SHA256 均通过。")


if __name__ == "__main__":
    main()
