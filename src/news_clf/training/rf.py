"""TF-IDF + 随机森林基线。"""

from __future__ import annotations

from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from news_clf.config import Settings, get_settings
from news_clf.data.preprocess import prepare_rf
from news_clf.training.common import compute_metrics, save_metrics, set_seed, setup_logger


def _load_stopwords(settings: Settings) -> Optional[list[str]]:
    if not settings.stopwords_file.exists():
        return None
    words = [
        line.strip()
        for line in settings.stopwords_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return words or None


def train(settings: Optional[Settings] = None) -> dict:
    settings = settings or get_settings()
    settings.ensure_dirs()
    set_seed(settings.seed)
    logger = setup_logger("train_rf", settings)

    # 1. 预处理数据不存在时自动生成
    processed = settings.processed_dir / "rf"
    if not (processed / "train.tsv").exists():
        logger.info("未找到分词数据，开始生成 ...")
        prepare_rf(settings)

    # 2. 读取分词结果
    train_df = pd.read_csv(processed / "train.tsv", sep="\t")
    test_df = pd.read_csv(processed / "test.tsv", sep="\t")
    logger.info("训练集 %d 条 / 测试集 %d 条", len(train_df), len(test_df))

    # 3. TF-IDF 特征
    stopwords = _load_stopwords(settings)
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    x_all = vectorizer.fit_transform(train_df["words"])
    logger.info("TF-IDF 特征维度: %d", x_all.shape[1])

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_all,
        train_df["label"],
        test_size=settings.rf_test_size,
        random_state=settings.seed,
        stratify=train_df["label"],
    )

    # 4. 训练
    model = RandomForestClassifier(
        n_estimators=settings.rf_n_estimators, n_jobs=-1, random_state=settings.seed
    )
    logger.info("随机森林开始训练（%d 棵树）...", settings.rf_n_estimators)
    model.fit(x_train, y_train)

    # 5. 评估：留出集 + 官方测试集
    holdout = compute_metrics(y_valid.tolist(), model.predict(x_valid).tolist())
    test_metrics = compute_metrics(
        test_df["label"].tolist(),
        model.predict(vectorizer.transform(test_df["words"])).tolist(),
    )
    logger.info("留出集: %s", holdout)
    logger.info("测试集: %s", test_metrics)

    # 6. 保存模型
    joblib.dump(vectorizer, settings.tfidf_path)
    joblib.dump(model, settings.rf_path)
    logger.info("模型已保存: %s / %s", settings.tfidf_path.name, settings.rf_path.name)

    save_metrics(
        "tfidf_rf",
        test_metrics,
        settings,
        extra={"holdout": holdout, "feature_dim": int(x_all.shape[1])},
    )
    return {"holdout": holdout, "test": test_metrics}
