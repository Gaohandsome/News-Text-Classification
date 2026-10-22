"""数据预处理：分词 / 分字，以及转换为各模型所需的输入格式。

- 随机森林：``text<TAB>words<TAB>label``（TSV，带表头）
- FastText：``__label__<类别> <token> <token> ...``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import jieba

from news_clf.config import Settings, get_settings
from news_clf.data.loader import SPLITS, read_examples, split_path

logger = logging.getLogger(__name__)

Granularity = Literal["word", "char"]


def _tokenizer(granularity: Granularity):
    if granularity == "word":
        return lambda text: jieba.lcut(text)
    if granularity == "char":
        return lambda text: list(text)
    raise ValueError(f"不支持的切分粒度: {granularity!r}，可选 'word' / 'char'")


def tokenize(text: str, granularity: Granularity = "word") -> list[str]:
    """按词或按字切分。"""
    return [token for token in _tokenizer(granularity)(text) if token.strip()]


def join_tokens(text: str, granularity: Granularity = "word") -> str:
    """切分后用空格拼接，供 TF-IDF / FastText 使用。"""
    return " ".join(tokenize(text, granularity))


def to_fasttext_line(text: str, label_name: str, granularity: Granularity = "word") -> str:
    return f"__label__{label_name} {join_tokens(text, granularity)}\n"


# --------------------------------------------------------------------- 随机森林
def prepare_rf(settings: Settings | None = None) -> dict[str, Path]:
    """生成随机森林用的分词数据（train / dev / test）。"""
    settings = settings or get_settings()
    settings.validate_data()
    labels = settings.labels
    out_dir = settings.processed_dir / "rf"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for split in SPLITS:
        dst = out_dir / f"{split}.tsv"
        examples = read_examples(split_path(split, settings), labels)
        with dst.open("w", encoding="utf-8", newline="") as handle:
            handle.write("text\twords\tlabel\n")
            for example in examples:
                text = example.text.replace("\t", " ").replace("\n", " ")
                words = join_tokens(example.text, "word").replace("\t", " ")
                handle.write(f"{text}\t{words}\t{example.label}\n")
        outputs[split] = dst
        logger.info("随机森林数据已生成: %s (%d 条)", dst, len(examples))
    return outputs


# --------------------------------------------------------------------- FastText
def prepare_fasttext(
    settings: Settings | None = None, granularity: Granularity = "word"
) -> dict[str, Path]:
    """生成 FastText 要求的 ``__label__xxx`` 格式数据。"""
    settings = settings or get_settings()
    settings.validate_data()
    labels = settings.labels
    out_dir = settings.processed_dir / "fasttext"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for split in SPLITS:
        dst = out_dir / f"{split}_{granularity}.txt"
        examples = read_examples(split_path(split, settings), labels)
        with dst.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(
                    to_fasttext_line(
                        example.text, labels[example.label], granularity=granularity
                    )
                )
        outputs[split] = dst
        logger.info("FastText(%s) 数据已生成: %s (%d 条)", granularity, dst, len(examples))
    return outputs


def prepare_all(
    settings: Settings | None = None, granularity: Granularity = "word"
) -> dict[str, dict[str, Path]]:
    """一次性准备好全部预处理数据。"""
    return {
        "rf": prepare_rf(settings),
        f"fasttext_{granularity}": prepare_fasttext(settings, granularity),
    }
