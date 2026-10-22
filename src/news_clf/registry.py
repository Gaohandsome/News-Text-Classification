"""模型注册表。

一处登记：模型的展示名、说明、产物路径、构造函数、所需环境变量。
Web 服务、CLI、评估脚本都从这里取模型，新增模型只需在 ``MODEL_SPECS`` 加一条。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Tuple

from news_clf.config import Settings, get_settings
from news_clf.inference.base import Predictor


def _fasttext_granularity() -> str:
    return os.getenv("NEWSCLF_FASTTEXT_GRANULARITY", "word")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    description: str
    artifacts: Callable[[Settings], Tuple[Path, ...]]
    factory: Callable[[Settings], Predictor]
    requires_env: Tuple[str, ...] = field(default=())


def _rf_factory(settings: Settings) -> Predictor:
    from news_clf.inference.classic import TfidfRfPredictor

    return TfidfRfPredictor(settings)


def _fasttext_factory(settings: Settings) -> Predictor:
    from news_clf.inference.classic import FastTextPredictor

    return FastTextPredictor(settings)


def _bert_factory(settings: Settings) -> Predictor:
    from news_clf.inference.torch_models import BertPredictor

    return BertPredictor(settings)


def _quantized_factory(settings: Settings) -> Predictor:
    from news_clf.inference.torch_models import QuantizedBertPredictor

    return QuantizedBertPredictor(settings)


def _distilled_factory(settings: Settings) -> Predictor:
    from news_clf.inference.torch_models import DistilledBiLSTMPredictor

    return DistilledBiLSTMPredictor(settings)


def _pruned_factory(settings: Settings) -> Predictor:
    from news_clf.inference.torch_models import PrunedBertPredictor

    return PrunedBertPredictor(settings)


def _llm_factory(settings: Settings) -> Predictor:
    from news_clf.inference.llm import LLMPredictor

    return LLMPredictor(settings)


MODEL_SPECS: Dict[str, ModelSpec] = {
    "tfidf-rf": ModelSpec(
        key="tfidf-rf",
        display_name="TF-IDF + 随机森林",
        family="传统机器学习",
        description="jieba 分词 + TF-IDF 特征 + 100 棵决策树的随机森林，训练快、可解释。",
        artifacts=lambda s: (s.tfidf_path, s.rf_path),
        factory=_rf_factory,
    ),
    "fasttext": ModelSpec(
        key="fasttext",
        display_name="FastText",
        family="浅层神经网络",
        description="词袋 + n-gram 的有监督文本分类，支持自动调参，CPU 上毫秒级推理。",
        artifacts=lambda s: (s.fasttext_path(_fasttext_granularity()),),
        factory=_fasttext_factory,
    ),
    "bert": ModelSpec(
        key="bert",
        display_name="BERT 微调",
        family="预训练语言模型",
        description="bert-base-chinese 全参数微调，本项目精度最高的单模型基线。",
        artifacts=lambda s: (s.bert_path,),
        factory=_bert_factory,
    ),
    "llm": ModelSpec(
        key="llm",
        display_name="LLM 零样本",
        family="大语言模型",
        description="通过 OpenAI 兼容接口调用 DeepSeek / GLM / 通义千问，无需训练，提示词约束输出。",
        artifacts=lambda _s: (),
        factory=_llm_factory,
    ),
    "bert-quantized": ModelSpec(
        key="bert-quantized",
        display_name="BERT 动态量化",
        family="模型压缩",
        description="对 nn.Linear 做 int8 动态量化，体积显著下降，仅用 CPU 推理。",
        artifacts=lambda s: (s.bert_quantized_path,),
        factory=_quantized_factory,
    ),
    "bilstm-distilled": ModelSpec(
        key="bilstm-distilled",
        display_name="BiLSTM 蒸馏",
        family="模型压缩",
        description="以 BERT 为教师做软标签蒸馏，把知识迁移到轻量 BiLSTM 学生模型。",
        artifacts=lambda s: (s.bilstm_path,),
        factory=_distilled_factory,
    ),
    "bert-pruned": ModelSpec(
        key="bert-pruned",
        display_name="BERT 权重剪枝",
        family="模型压缩",
        description="对注意力 query 权重做全局非结构化 L1 剪枝并固化稀疏结构。",
        artifacts=lambda s: (s.bert_pruned_path,),
        factory=_pruned_factory,
    ),
}

MODEL_ALIASES: Dict[str, str] = {
    "rf": "tfidf-rf",
    "random-forest": "tfidf-rf",
    "randomforest": "tfidf-rf",
    "sklearn": "tfidf-rf",
    "ft": "fasttext",
    "fast-text": "fasttext",
    "bert-base": "bert",
    "finetune": "bert",
    "quantized": "bert-quantized",
    "quantization": "bert-quantized",
    "int8": "bert-quantized",
    "distilled": "bilstm-distilled",
    "distillation": "bilstm-distilled",
    "bilstm": "bilstm-distilled",
    "pruned": "bert-pruned",
    "pruning": "bert-pruned",
}


def resolve_key(name: str) -> str:
    """把别名/大小写统一成注册表中的 key。"""
    if not name:
        raise ValueError("模型名称不能为空")
    key = name.strip().lower()
    key = MODEL_ALIASES.get(key, key)
    if key not in MODEL_SPECS:
        raise ValueError(
            f"未知模型 {name!r}，可选: {', '.join(MODEL_SPECS)}"
            f"（别名: {', '.join(sorted(MODEL_ALIASES))}）"
        )
    return key


def model_availability(spec: ModelSpec, settings: Settings) -> tuple[bool, str]:
    missing_artifacts = [p for p in spec.artifacts(settings) if not p.exists()]
    if missing_artifacts:
        names = ", ".join(p.name for p in missing_artifacts)
        return False, f"缺少模型文件: {names}"

    if spec.key == "llm":
        # LLM 是否需要 Key 取决于所选供应商，这里按当前 provider 判断
        from news_clf.models.llm import PROVIDERS

        provider = PROVIDERS.get(settings.llm_provider)
        if provider is None:
            return False, f"未知 LLM 供应商: {settings.llm_provider}"
        if not os.getenv(provider.api_key_env):
            return False, f"缺少环境变量: {provider.api_key_env}"

    missing_env = [name for name in spec.requires_env if not os.getenv(name)]
    if missing_env:
        return False, f"缺少环境变量: {', '.join(missing_env)}"

    return True, "ready"


def list_models(settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    items = []
    for spec in MODEL_SPECS.values():
        available, status = model_availability(spec, settings)
        items.append(
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "family": spec.family,
                "description": spec.description,
                "available": available,
                "status": status,
            }
        )
    return items


def available_keys(settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    return [item["key"] for item in list_models(settings) if item["available"]]


def first_available(settings: Settings | None = None) -> str | None:
    keys = available_keys(settings)
    return keys[0] if keys else None


_CACHE: Dict[str, Predictor] = {}


def create_predictor(
    name: str, settings: Settings | None = None, use_cache: bool = True
) -> Predictor:
    """按名称创建（并缓存）预测器。"""
    settings = settings or get_settings()
    key = resolve_key(name)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    spec = MODEL_SPECS[key]
    available, status = model_availability(spec, settings)
    if not available:
        raise FileNotFoundError(
            f"模型 {spec.display_name} 尚不可用（{status}）。\n"
            "请先训练/压缩对应模型，或检查 .env 中的 API Key。"
        )

    predictor = spec.factory(settings)
    if use_cache:
        _CACHE[key] = predictor
    return predictor


def clear_cache() -> None:
    _CACHE.clear()
