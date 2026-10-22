"""训练与评估的公共工具：随机种子、设备、指标、日志、结果落盘。"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from news_clf.config import Settings, get_settings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(settings: Optional[Settings] = None):
    import torch

    settings = settings or get_settings()
    return torch.device(settings.device)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """多分类宏平均指标。"""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "num_samples": len(y_true),
    }


def evaluate_torch(model, dataloader, device, description: str = "eval") -> dict:
    """在给定 DataLoader 上评估 torch 模型。"""
    import torch
    from tqdm import tqdm

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        # build_dataloaders 返回的批次是 (BatchEncoding, labels)
        for features, labels in tqdm(dataloader, desc=description, leave=False):
            features = features.to(device)
            labels = labels.to(device)
            logits = model(**features)
            y_true.extend(labels.tolist())
            y_pred.extend(torch.argmax(logits, dim=-1).tolist())

    model.train()
    return compute_metrics(y_true, y_pred)


def setup_logger(name: str, settings: Optional[Settings] = None) -> logging.Logger:
    """控制台 + 文件双输出。"""
    settings = settings or get_settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if getattr(logger, "_news_clf_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = settings.logs_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger._news_clf_configured = True  # type: ignore[attr-defined]
    return logger


def save_metrics(
    name: str,
    metrics: dict,
    settings: Optional[Settings] = None,
    extra: Optional[dict] = None,
) -> Path:
    """把指标写入 ``outputs/reports/<name>_metrics.json``。"""
    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": settings.as_dict(),
    }
    if extra:
        payload.update(extra)
    payload["metrics"] = metrics

    path = settings.reports_dir / f"{name}_metrics.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


class Timer:
    """``with Timer() as t: ...`` 之后可用 ``t.elapsed``。"""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self.elapsed = round(time.perf_counter() - self.start, 2)
