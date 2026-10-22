"""知识蒸馏：用 BERT 教师模型指导 BiLSTM 学生模型。

损失 = alpha * T^2 * KL(学生软标签 || 教师软标签) + (1 - alpha) * 硬标签交叉熵
其中硬标签采用教师模型的预测结果（也可以改用人工标注，见 hard_from_gold 参数）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from news_clf.config import Settings, get_settings
from news_clf.data.dataset import build_dataloaders
from news_clf.models.bert import build_classifier, load_tokenizer
from news_clf.models.bilstm import build_bilstm
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
    eval_steps: int = 200,
    log_steps: int = 10,
    hard_from_gold: bool = False,
) -> dict:
    settings = settings or get_settings()
    settings.ensure_dirs()
    set_seed(settings.seed)
    logger = setup_logger("train_distillation", settings)

    if not settings.bert_path.exists():
        raise FileNotFoundError(
            f"未找到教师模型 {settings.bert_path}，请先执行: news-clf train --model bert"
        )

    device = resolve_device(settings)
    logger.info("运行设备: %s", device)

    tokenizer = load_tokenizer(settings)
    loaders = build_dataloaders(settings, tokenizer)

    # 教师：冻结参数，只做前向
    teacher = build_classifier(settings, pretrained=False).to(device)
    teacher.load_state_dict(torch.load(settings.bert_path, map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    # 学生
    student = build_bilstm(settings).to(device)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info("学生模型可训练参数量: %.2fM", trainable / 1e6)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=settings.lstm_lr, betas=(0.9, 0.999))
    alpha, temperature = settings.distill_alpha, settings.distill_temperature

    epochs = epochs or settings.bert_epochs
    best_f1 = -1.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        student.train()
        window_loss, window_count = 0.0, 0
        y_true, y_pred = [], []

        progress = tqdm(
            enumerate(loaders.train, start=1),
            total=len(loaders.train),
            desc=f"distill epoch {epoch}/{epochs}",
        )
        for step, (features, labels) in progress:
            features = features.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(**features)
                teacher_labels = torch.argmax(teacher_logits, dim=-1)

            student_logits = student(**features)
            hard_target = labels if hard_from_gold else teacher_labels
            hard_loss = loss_fn(student_logits, hard_target)

            teacher_soft = F.log_softmax(teacher_logits / temperature, dim=-1)
            student_soft = F.log_softmax(student_logits / temperature, dim=-1)
            soft_loss = F.kl_div(
                student_soft, teacher_soft, reduction="batchmean", log_target=True
            )
            loss = alpha * (temperature**2) * soft_loss + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            window_loss += loss.item()
            window_count += 1
            y_true.extend(labels.tolist())
            y_pred.extend(torch.argmax(student_logits, dim=-1).tolist())

            if step % log_steps == 0 or step == len(loaders.train):
                metrics = compute_metrics(y_true, y_pred)
                avg_loss = window_loss / max(window_count, 1)
                progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{metrics['accuracy']:.4f}")
                logger.info(
                    "轮次 %d | 批次 %d/%d | 损失 %.4f | 准确率 %.4f | F1 %.4f",
                    epoch,
                    step,
                    len(loaders.train),
                    avg_loss,
                    metrics["accuracy"],
                    metrics["f1_macro"],
                )
                window_loss, window_count = 0.0, 0
                y_true, y_pred = [], []

            if step % eval_steps == 0 or step == len(loaders.train):
                dev_metrics = evaluate_torch(student, loaders.dev, device, "dev")
                logger.info(
                    "验证 | 准确率 %.4f | F1 %.4f",
                    dev_metrics["accuracy"],
                    dev_metrics["f1_macro"],
                )
                history.append({"epoch": epoch, "step": step, **dev_metrics})
                if dev_metrics["f1_macro"] > best_f1:
                    best_f1 = dev_metrics["f1_macro"]
                    torch.save(student.state_dict(), settings.bilstm_path)
                    logger.info("验证集 F1 创新高 %.4f，学生模型已保存", best_f1)

    if settings.bilstm_path.exists():
        student.load_state_dict(torch.load(settings.bilstm_path, map_location=device))

    test_metrics = evaluate_torch(student, loaders.test, device, "test")
    logger.info("测试集: %s", test_metrics)

    save_metrics(
        "bilstm_distilled",
        test_metrics,
        settings,
        extra={
            "best_dev_f1": round(best_f1, 4),
            "alpha": alpha,
            "temperature": temperature,
            "trainable_params": trainable,
            "epochs": epochs,
        },
    )
    return test_metrics
