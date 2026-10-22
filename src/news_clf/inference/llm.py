"""LLM 推理封装。"""

from __future__ import annotations

from typing import Dict

from news_clf.config import Settings
from news_clf.inference.base import Predictor
from news_clf.models.llm import LLMClassifier


class LLMPredictor(Predictor):
    key = "llm"
    display_name = "LLM 零样本分类"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.classifier = LLMClassifier(self.settings)

    def _predict(self, text: str) -> Dict[str, float]:
        label, _raw = self.classifier.classify(text)
        # LLM 只给出类别，不返回概率，这里用 one-hot 表达
        scores = {candidate: 0.0 for candidate in self.labels}
        scores[label] = 1.0
        return scores
