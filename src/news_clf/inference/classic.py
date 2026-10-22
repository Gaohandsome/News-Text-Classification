"""传统机器学习模型的推理实现（随机森林 / FastText）。"""

from __future__ import annotations

import os
from typing import Dict

import jieba

from news_clf.config import Settings
from news_clf.data.preprocess import join_tokens
from news_clf.inference.base import Predictor


class TfidfRfPredictor(Predictor):
    key = "tfidf-rf"
    display_name = "TF-IDF + 随机森林"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        import joblib

        for path in (self.settings.tfidf_path, self.settings.rf_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"缺少模型文件 {path}，请先训练: news-clf train --model tfidf-rf"
                )
        self.vectorizer = joblib.load(self.settings.tfidf_path)
        self.model = joblib.load(self.settings.rf_path)

    def _predict(self, text: str) -> Dict[str, float]:
        words = " ".join(jieba.lcut(text))
        features = self.vectorizer.transform([words])

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            classes = self.model.classes_
            return {
                self.labels[int(cls)]: float(prob)
                for cls, prob in zip(classes, probabilities)
            }

        # 没有概率输出时退化为 one-hot
        predicted = int(self.model.predict(features)[0])
        scores = {label: 0.0 for label in self.labels}
        scores[self.labels[predicted]] = 1.0
        return scores


class FastTextPredictor(Predictor):
    key = "fasttext"
    display_name = "FastText"

    def __init__(self, settings: Settings | None = None, granularity: str | None = None) -> None:
        super().__init__(settings)
        import fasttext

        self.granularity = granularity or os.getenv("NEWSCLF_FASTTEXT_GRANULARITY", "word")
        self.path = self.settings.fasttext_path(self.granularity)
        if not self.path.exists():
            raise FileNotFoundError(
                f"缺少模型文件 {self.path}，请先训练: news-clf train --model fasttext"
            )
        self.model = fasttext.load_model(str(self.path))

    def _predict(self, text: str) -> Dict[str, float]:
        tokens = join_tokens(text, self.granularity)  # type: ignore[arg-type]
        labels, probabilities = self.model.predict(tokens, k=len(self.labels))
        return {
            label.replace("__label__", ""): float(prob)
            for label, prob in zip(labels, probabilities)
        }
