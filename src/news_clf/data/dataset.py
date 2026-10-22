"""PyTorch 数据集与 DataLoader 构建。

相比原实现的两点修正：
1. 统一返回具名结构 ``DataLoaders(train, dev, test)``，避免 "train, test, dev" 顺序错乱；
2. 修复了原代码 ``batch_x.to(device)`` 未接收返回值、数据其实没上 GPU 的问题
   （BatchEncoding.to() 是原地修改，但仍建议显式调用并保留返回值语义）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from news_clf.config import Settings, get_settings
from news_clf.data.loader import Example, read_split


@dataclass(slots=True)
class DataLoaders:
    train: DataLoader
    dev: DataLoader
    test: DataLoader

    def as_tuple(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        return self.train, self.dev, self.test


class NewsDataset(Dataset):
    def __init__(self, examples: Sequence[Example]) -> None:
        self.examples: List[Example] = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return example.text, example.label


def build_collate_fn(tokenizer, max_len: int) -> Callable:
    """把一批 ``(text, label)`` 转成 ``BatchEncoding + LongTensor``。"""

    def collate(batch):
        texts, labels = zip(*batch)
        features = tokenizer(
            list(texts),
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return features, torch.tensor(labels, dtype=torch.long)

    return collate


def build_dataloaders(
    settings: Settings | None = None,
    tokenizer=None,
    batch_size: int | None = None,
) -> DataLoaders:
    settings = settings or get_settings()
    if tokenizer is None:
        from news_clf.models.bert import load_tokenizer

        tokenizer = load_tokenizer(settings)

    batch_size = batch_size or settings.batch_size
    collate = build_collate_fn(tokenizer, settings.max_len)
    common = dict(
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=settings.num_workers,
        pin_memory=settings.device.startswith("cuda"),
    )

    train = DataLoader(
        NewsDataset(read_split("train", settings)), shuffle=True, **common
    )
    dev = DataLoader(NewsDataset(read_split("dev", settings)), shuffle=False, **common)
    test = DataLoader(NewsDataset(read_split("test", settings)), shuffle=False, **common)
    return DataLoaders(train=train, dev=dev, test=test)
