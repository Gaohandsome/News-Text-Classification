"""下载 bert-base-chinese 预训练权重到 ``pretrained/`` 目录。

优先 ModelScope（国内下载快），失败时回退 HuggingFace。

用法::

    python scripts/download_pretrained.py
    python scripts/download_pretrained.py --source huggingface
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from news_clf.config import get_settings  # noqa: E402

MODEL_ID = "google-bert/bert-base-chinese"
REQUIRED_FILES = ("config.json", "vocab.txt", "tokenizer_config.json")


def _has_weights(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin")
    )


def download_modelscope(target: Path) -> None:
    from modelscope import snapshot_download

    snapshot_download(MODEL_ID, local_dir=str(target))


def download_huggingface(target: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_ID, local_dir=str(target))


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 BERT 预训练权重")
    parser.add_argument(
        "--source",
        choices=["auto", "modelscope", "huggingface"],
        default="auto",
        help="下载源，默认先试 ModelScope 再回退 HuggingFace",
    )
    parser.add_argument("--output", type=Path, default=None, help="目标目录")
    args = parser.parse_args()

    settings = get_settings()
    target = args.output or settings.bert_pretrained_dir
    target.mkdir(parents=True, exist_ok=True)

    if _has_weights(target) and all((target / f).exists() for f in REQUIRED_FILES):
        print(f"已存在完整权重，跳过下载: {target}")
        return 0

    sources = (
        ["modelscope", "huggingface"] if args.source == "auto" else [args.source]
    )
    errors: list[str] = []

    for source in sources:
        try:
            print(f"正在从 {source} 下载 {MODEL_ID} ...")
            if source == "modelscope":
                download_modelscope(target)
            else:
                download_huggingface(target)

            missing = [f for f in REQUIRED_FILES if not (target / f).exists()]
            if missing or not _has_weights(target):
                raise RuntimeError(f"下载后仍缺少文件: {missing or ['权重文件']}")

            print(f"下载完成: {target}")
            return 0
        except Exception as exc:  # noqa: BLE001 - 需要尝试下一个源
            errors.append(f"{source}: {exc}")
            print(f"从 {source} 下载失败: {exc}", file=sys.stderr)

    print("\n全部下载源均失败：", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    print(
        "\n可手动下载 bert-base-chinese（config.json / vocab.txt / tokenizer*.json / 权重）"
        f"\n并放到: {target}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
