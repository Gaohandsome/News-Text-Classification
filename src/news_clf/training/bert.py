"""BERT 微调训练。"""

from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm

from news_clf.config import Settings, get_settings
from news_clf.data.dataset import build_dataloaders
from news_clf.models.bert import build_classifier, load_tokenizer
from news_clf.training.common import (
    compute_metrics,
    evaluate_torch,
    resolve_device,
    save_metrics,
    set_seed,
    setup_logger,
)


def train(
    settings: Optional[Settings] = None,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    eval_steps: int = 200,
    log_steps: int = 10,
    limit_batches: Optional[int] = None,
) -> dict:
    """训练 BERT 分类模型，按验证集 f1 保存最优权重。

    :param limit_batches: 每轮只跑前 N 个 batch，仅用于冒烟测试
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    set_seed(settings.seed)
    logger = setup_logger("train_bert", settings)

    device = resolve_device(settings)
    logger.info("运行设备: %s", device)

    tokenizer = load_tokenizer(settings)
    loaders = build_dataloaders(settings, tokenizer)

    model = build_classifier(settings, pretrained=True).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr or settings.bert_lr, betas=(0.9, 0.999)
    )

    epochs = epochs or settings.bert_epochs
    best_f1 = -1.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, window_loss, window_count = 0.0, 0.0, 0
        y_true: list[int] = []
        y_pred: list[int] = []

        total_steps = len(loaders.train)
        if limit_batches:
            total_steps = min(total_steps, limit_batches)

        progress = tqdm(
            enumerate(loaders.train, start=1),
            total=total_steps,
            desc=f"epoch {epoch}/{epochs}",
        )
        for step, (features, labels) in progress:
            if limit_batches and step > limit_batches:
                break

            features = features.to(device)
            labels = labels.to(device)

            logits = model(**features)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            window_loss += loss.item()
            window_count += 1
            y_true.extend(labels.tolist())
            y_pred.extend(torch.argmax(logits, dim=-1).tolist())

            # --- 训练日志 ---
            if step % log_steps == 0 or step == total_steps:
                window_metrics = compute_metrics(y_true, y_pred)
                avg_loss = window_loss / max(window_count, 1)
                progress.set_postfix(
                    loss=f"{avg_loss:.4f}", acc=f"{window_metrics['accuracy']:.4f}"
                )
                logger.info(
                    "轮次 %d | 批次 %d/%d | 损失 %.4f | 准确率 %.4f | 精确率 %.4f | 召回率 %.4f | F1 %.4f",
                    epoch,
                    step,
                    total_steps,
                    avg_loss,
                    window_metrics["accuracy"],
                    window_metrics["precision_macro"],
                    window_metrics["recall_macro"],
                    window_metrics["f1_macro"],
                )
                window_loss, window_count = 0.0, 0
                y_true, y_pred = [], []

            # --- 验证与保存 ---
            if step % eval_steps == 0 or step == total_steps:
                dev_metrics = evaluate_torch(model, loaders.dev, device, "dev")
                logger.info(
                    "验证 | 准确率 %.4f | 精确率 %.4f | 召回率 %.4f | F1 %.4f",
                    dev_metrics["accuracy"],
                    dev_metrics["precision_macro"],
                    dev_metrics["recall_macro"],
                    dev_metrics["f1_macro"],
                )
                history.append({"epoch": epoch, "step": step, **dev_metrics})
                if dev_metrics["f1_macro"] > best_f1:
                    best_f1 = dev_metrics["f1_macro"]
                    torch.save(model.state_dict(), settings.bert_path)
                    logger.info("验证集 F1 创新高 %.4f，模型已保存", best_f1)

    # --- 用最优权重在测试集上评估 ---
    if settings.bert_path.exists():
        model.load_state_dict(torch.load(settings.bert_path, map_location=device))

    test_metrics = evaluate_torch(model, loaders.test, device, "test")
    logger.info("测试集: %s", test_metrics)

    save_metrics(
        "bert",
        test_metrics,
        settings,
        extra={
            "best_dev_f1": round(best_f1, 4),
            "epochs": epochs,
            "lr": lr or settings.bert_lr,
            "batch_size": settings.batch_size,
            "max_len": settings.max_len,
            "history": history,
        },
    )
    return test_metrics
