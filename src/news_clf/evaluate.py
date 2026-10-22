"""批量评估：任何注册模型都能用同一套流程在 dev/test 上打分。"""

from __future__ import annotations

from typing import Optional

from tqdm import tqdm

from news_clf.config import Settings, get_settings
from news_clf.data.loader import read_split
from news_clf.labels import label2id
from news_clf.registry import create_predictor, resolve_key
from news_clf.training.common import compute_metrics, save_metrics


def evaluate(
    model: str,
    split: str = "test",
    limit: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> dict:
    """在指定数据集划分上评估模型，返回指标字典并落盘。"""
    settings = settings or get_settings()
    key = resolve_key(model)
    predictor = create_predictor(key, settings)

    examples = read_split(split, settings)
    if limit:
        examples = examples[:limit]

    mapping = label2id(settings.labels)
    y_true: list[int] = []
    y_pred: list[int] = []
    failures = 0

    for example in tqdm(examples, desc=f"{key} @ {split}"):
        try:
            prediction = predictor.predict(example.text)
        except Exception:  # 单条失败不应中断整体评估（例如 LLM 限流）
            failures += 1
            continue
        y_true.append(example.label)
        y_pred.append(mapping[prediction.label])

    if not y_true:
        raise RuntimeError("所有样本都预测失败，请检查模型与 API 配置。")

    metrics = compute_metrics(y_true, y_pred)
    if failures:
        metrics["failed"] = failures

    save_metrics(
        f"{key}_{split}",
        metrics,
        settings,
        extra={"split": split, "limit": limit, "failures": failures},
    )
    return metrics
