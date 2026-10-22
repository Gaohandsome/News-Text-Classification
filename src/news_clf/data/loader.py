"""数据读取。

原始数据是 ``text<TAB>label_id`` 的纯文本，这里统一用 ``rsplit("\\t", 1)`` 解析，
即使正文中出现制表符也不会把标签切错（原实现用 ``split("\\t")`` 会直接报错）。
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from news_clf.config import Settings, get_settings

logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "dev", "test")


@dataclass(frozen=True, slots=True)
class Example:
    """一条样本。"""

    text: str
    label: int


def split_path(split: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    mapping = {
        "train": settings.train_file,
        "dev": settings.dev_file,
        "test": settings.test_file,
    }
    if split not in mapping:
        raise ValueError(f"未知的数据集划分: {split!r}，可选 {SPLITS}")
    return mapping[split]


def read_examples(path: Path, labels: Sequence[str] | None = None) -> List[Example]:
    """读取一个数据文件，返回样本列表。

    :param path: 数据文件路径
    :param labels: 类别序列，传入时会校验标签 id 是否越界
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    num_labels = len(labels) if labels is not None else None
    examples: List[Example] = []
    skipped = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            try:
                text, label_str = line.rsplit("\t", 1)
                label = int(label_str)
            except ValueError:
                skipped += 1
                logger.warning("跳过格式错误的第 %d 行: %r", line_no, line[:80])
                continue
            if num_labels is not None and not 0 <= label < num_labels:
                skipped += 1
                logger.warning("跳过标签越界的第 %d 行: 标签=%d", line_no, label)
                continue
            examples.append(Example(text=text.strip(), label=label))

    if skipped:
        logger.warning("%s 共跳过 %d 行异常数据", path.name, skipped)
    if not examples:
        raise ValueError(f"数据文件为空或全部解析失败: {path}")
    return examples


def read_split(split: str, settings: Settings | None = None) -> List[Example]:
    settings = settings or get_settings()
    settings.validate_data()
    return read_examples(split_path(split, settings), settings.labels)


def class_counts(
    examples: Sequence[Example], labels: Sequence[str] | None = None
) -> dict[str, int]:
    """按类别名统计样本数，便于检查类别是否均衡。"""
    if labels is None:
        from news_clf.config import get_settings

        labels = get_settings().labels

    counter = Counter(example.label for example in examples)
    return {label: counter.get(idx, 0) for idx, label in enumerate(labels)}
