"""FastText 有监督文本分类。

注意：fasttext 在 Windows 上打开含中文的路径会失败，因此本项目把整个工程
放在纯 ASCII 路径下即可避免该问题（原项目路径含中文时需要额外处理）。
"""

from __future__ import annotations

from typing import Optional

from news_clf.config import Settings, get_settings
from news_clf.data.preprocess import Granularity, prepare_fasttext
from news_clf.training.common import save_metrics, set_seed, setup_logger


def train(
    settings: Optional[Settings] = None,
    granularity: Granularity = "word",
    autotune: Optional[bool] = None,
) -> dict:
    settings = settings or get_settings()
    settings.ensure_dirs()
    set_seed(settings.seed)
    logger = setup_logger("train_fasttext", settings)

    import fasttext

    autotune = settings.fasttext_autotune if autotune is None else autotune

    # 1. 预处理
    paths = prepare_fasttext(settings, granularity)

    # 2. 训练
    if autotune:
        logger.info(
            "FastText 自动调参训练（%ss，验证集 %s）...",
            settings.fasttext_autotune_duration,
            paths["dev"].name,
        )
        model = fasttext.train_supervised(
            input=str(paths["train"]),
            autotuneValidationFile=str(paths["dev"]),
            autotuneDuration=settings.fasttext_autotune_duration,
            verbose=3,
        )
    else:
        logger.info("FastText 默认参数训练 ...")
        model = fasttext.train_supervised(
            input=str(paths["train"]),
            lr=settings.fasttext_lr,
            epoch=settings.fasttext_epoch,
            wordNgrams=settings.fasttext_word_ngrams,
            verbose=2,
        )

    save_path = settings.fasttext_path(granularity)
    model.save_model(str(save_path))
    logger.info("模型已保存: %s", save_path.name)

    # 3. 评估（fasttext.test 返回 (样本数, 精确率, 召回率)）
    num_samples, precision, recall = model.test(str(paths["test"]))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics = {
        "accuracy": round(float(precision), 4),  # 单标签分类下 precision == accuracy
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "num_samples": int(num_samples),
    }
    logger.info("测试集: %s", metrics)

    save_metrics(
        f"fasttext_{granularity}",
        metrics,
        settings,
        extra={"granularity": granularity, "autotune": autotune},
    )
    return metrics
