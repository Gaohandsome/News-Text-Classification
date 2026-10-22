"""模型压缩：动态量化 与 全局非结构化剪枝。

量化流程（与原项目保持一致）：加载 float 权重 -> ``quantize_dynamic`` ->
保存量化后的 state_dict。推理侧必须先量化结构再 load_state_dict，
因为量化后的 Linear 使用打包参数，键名与 float 模型不同。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn.utils.prune import L1Unstructured, global_unstructured, remove

from news_clf.config import Settings, get_settings
from news_clf.data.dataset import build_dataloaders
from news_clf.models.bert import build_classifier, load_tokenizer
from news_clf.training.common import evaluate_torch, save_metrics, set_seed, setup_logger


def _param_bytes(model: torch.nn.Module) -> int:
    total = 0
    for tensor in model.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel() * tensor.element_size()
    return total


def quantize(settings: Optional[Settings] = None) -> dict:
    settings = settings or get_settings()
    settings.ensure_dirs()
    set_seed(settings.seed)
    logger = setup_logger("compress_quantize", settings)

    if not settings.bert_path.exists():
        raise FileNotFoundError(
            f"未找到 {settings.bert_path}，请先训练 BERT: news-clf train --model bert"
        )

    tokenizer = load_tokenizer(settings)
    loaders = build_dataloaders(settings, tokenizer)
    cpu = torch.device("cpu")  # 动态量化只支持 CPU

    model = build_classifier(settings, pretrained=False)
    model.load_state_dict(torch.load(settings.bert_path, map_location="cpu"))
    model.eval()

    before = evaluate_torch(model, loaders.dev, cpu, "dev-float32")
    logger.info("量化前: %s", before)

    quantized = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )
    quantized.eval()
    after = evaluate_torch(quantized, loaders.dev, cpu, "dev-int8")
    logger.info("量化后: %s", after)

    torch.save(quantized.state_dict(), settings.bert_quantized_path)

    float_mb = _param_bytes(model) / 1024**2
    quant_mb = settings.bert_quantized_path.stat().st_size / 1024**2
    logger.info(
        "模型体积: %.1f MB -> %.1f MB（压缩至 %.1f%%）",
        float_mb,
        quant_mb,
        quant_mb / float_mb * 100,
    )

    metrics = {
        "accuracy": after["accuracy"],
        "f1_macro": after["f1_macro"],
        "size_mb_before": round(float_mb, 1),
        "size_mb_after": round(quant_mb, 1),
        "compression_ratio": round(quant_mb / float_mb, 3),
    }
    save_metrics(
        "bert_quantized",
        metrics,
        settings,
        extra={"before": before, "after": after, "method": "dynamic int8 (nn.Linear)"},
    )
    return metrics


def prune(
    settings: Optional[Settings] = None, amount: Optional[float] = None
) -> dict:
    settings = settings or get_settings()
    settings.ensure_dirs()
    set_seed(settings.seed)
    logger = setup_logger("compress_prune", settings)

    if not settings.bert_path.exists():
        raise FileNotFoundError(
            f"未找到 {settings.bert_path}，请先训练 BERT: news-clf train --model bert"
        )

    amount = settings.prune_amount if amount is None else amount
    device = torch.device(settings.device)
    tokenizer = load_tokenizer(settings)
    loaders = build_dataloaders(settings, tokenizer)

    model = build_classifier(settings, pretrained=False).to(device)
    model.load_state_dict(torch.load(settings.bert_path, map_location=device))
    model.eval()

    def query_sparsity() -> float:
        zero = total = 0
        for layer in model.bert.encoder.layer:
            weight = layer.attention.self.query.weight
            total += weight.numel()
            zero += int((weight == 0).sum())
        return zero / total if total else 0.0

    before = evaluate_torch(model, loaders.dev, device, "dev-before")
    logger.info("剪枝前稀疏度 %.2f%% | %s", query_sparsity() * 100, before)

    targets = [
        (layer.attention.self.query, "weight") for layer in model.bert.encoder.layer
    ]
    global_unstructured(parameters=targets, pruning_method=L1Unstructured, amount=amount)
    for module, name in targets:
        remove(module, name)
    logger.info("已将 query 权重按 L1 幅度剪掉 %.0f%% 并固化", amount * 100)

    after = evaluate_torch(model, loaders.dev, device, "dev-after")
    sparsity = query_sparsity()
    logger.info("剪枝后稀疏度 %.2f%% | %s", sparsity * 100, after)

    torch.save(model.state_dict(), settings.bert_pruned_path)

    metrics = {
        "accuracy": after["accuracy"],
        "f1_macro": after["f1_macro"],
        "sparsity": round(sparsity, 4),
        "prune_amount": amount,
    }
    save_metrics(
        "bert_pruned",
        metrics,
        settings,
        extra={"before": before, "after": after, "method": "global unstructured L1"},
    )
    return metrics
