"""PyTorch 模型推理：BERT / 量化 BERT / 蒸馏 BiLSTM / 剪枝 BERT。

四者共用同一套 tokenizer 与 predict 流程，差异只在如何构建并加载网络。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from news_clf.config import Settings
from news_clf.inference.base import Predictor


class TorchTextPredictor(Predictor):
    checkpoint: Path
    force_cpu: bool = False

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.device = torch.device("cpu" if self.force_cpu else self.settings.device)

        from news_clf.models.bert import load_tokenizer

        self.tokenizer = load_tokenizer(self.settings)
        self.model = self._load_model()
        self.model.eval()

    # ------------------------------------------------------------------ 子类实现
    def _build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def _load_state_dict(self) -> dict:
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"缺少模型文件 {self.checkpoint}，请先完成对应训练/压缩步骤。"
            )
        return torch.load(self.checkpoint, map_location="cpu")

    # ------------------------------------------------------------------ 加载
    def _load_model(self) -> torch.nn.Module:
        model = self._build_model()
        model.load_state_dict(self._load_state_dict())
        return model.to(self.device)

    # ------------------------------------------------------------------ 推理
    def _predict(self, text: str) -> Dict[str, float]:
        batch = self.tokenizer(
            text,
            max_length=self.settings.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.model(**batch)
            probabilities = torch.softmax(logits, dim=-1)[0].tolist()
        return {label: float(prob) for label, prob in zip(self.labels, probabilities)}


class BertPredictor(TorchTextPredictor):
    key = "bert"
    display_name = "BERT 微调"

    @property
    def checkpoint(self) -> Path:  # type: ignore[override]
        return self.settings.bert_path

    def _build_model(self):
        from news_clf.models.bert import build_classifier

        return build_classifier(self.settings, pretrained=False)


class QuantizedBertPredictor(TorchTextPredictor):
    key = "bert-quantized"
    display_name = "BERT + 动态量化 (int8)"
    force_cpu = True

    @property
    def checkpoint(self) -> Path:  # type: ignore[override]
        return self.settings.bert_quantized_path

    def _build_model(self):
        from news_clf.models.bert import build_classifier

        float_model = build_classifier(self.settings, pretrained=False)
        # 必须先把结构量化，才能加载量化后的 state_dict
        return torch.quantization.quantize_dynamic(
            float_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
        )


class PrunedBertPredictor(BertPredictor):
    key = "bert-pruned"
    display_name = "BERT + 权重剪枝"

    @property
    def checkpoint(self) -> Path:  # type: ignore[override]
        return self.settings.bert_pruned_path


class DistilledBiLSTMPredictor(TorchTextPredictor):
    key = "bilstm-distilled"
    display_name = "BiLSTM（BERT 蒸馏）"

    @property
    def checkpoint(self) -> Path:  # type: ignore[override]
        return self.settings.bilstm_path

    def _build_model(self):
        from news_clf.models.bilstm import build_bilstm

        return build_bilstm(self.settings)
