"""统一推理接口。

所有模型（传统 ML / 深度学习 / LLM）都实现同一个 ``Predictor`` 协议，
Web 服务、CLI、批量评估因此不必关心底层差异。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from news_clf.config import Settings, get_settings


@dataclass(slots=True)
class Prediction:
    label: str
    confidence: float
    scores: Dict[str, float]
    model: str
    latency_ms: float = 0.0

    def top(self, k: int = 3) -> List[tuple[str, float]]:
        ordered = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        return ordered[:k]

    def to_dict(self, top_k: int = 0) -> dict:
        scores = dict(sorted(self.scores.items(), key=lambda item: item[1], reverse=True))
        if top_k > 0:
            scores = dict(list(scores.items())[:top_k])
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "scores": {key: round(value, 4) for key, value in scores.items()},
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
        }


class Predictor(ABC):
    """所有分类器的公共父类。"""

    key: str = ""
    display_name: str = ""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.labels: Sequence[str] = self.settings.labels

    @abstractmethod
    def _predict(self, text: str) -> Dict[str, float]:
        """返回 ``{类别: 概率}``，概率之和应为 1（无概率信息时可用 one-hot）。"""

    def predict(self, text: str) -> Prediction:
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        started = time.perf_counter()
        scores = self._predict(text.strip())
        latency = (time.perf_counter() - started) * 1000

        # 只保留标签文件里定义的类别，避免模型输出的脏标签泄漏到上层
        scores = {label: float(scores.get(label, 0.0)) for label in self.labels}
        label = max(scores, key=scores.get)
        return Prediction(
            label=label,
            confidence=scores[label],
            scores=scores,
            model=self.key,
            latency_ms=latency,
        )

    def predict_batch(self, texts: Iterable[str]) -> List[Prediction]:
        return [self.predict(text) for text in texts]
