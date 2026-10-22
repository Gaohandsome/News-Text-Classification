"""全局配置。

设计原则：所有路径与超参数只在这里定义一次，其它模块一律从 Settings 取值。
原始项目中每个子目录各复制了一份 Config，且把 ``F:\\...`` 写死在代码里，
这里改为基于项目根目录自动推导，因此项目可以放在任意盘符、直接 clone 运行。

配置优先级：环境变量 > 默认值。项目根目录下的 ``.env`` 会在导入时自动加载。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

from news_clf.labels import read_labels

try:  # 允许在未安装可选依赖时仍然可以导入本模块
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 依赖缺失时的兜底

    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore[misc]
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """项目配置快照。"""

    # ---------------------------------------------------------------- 基础
    root: Path = PROJECT_ROOT
    seed: int = field(default_factory=lambda: _env_int("NEWSCLF_SEED", 42))
    num_workers: int = field(default_factory=lambda: _env_int("NEWSCLF_NUM_WORKERS", 0))

    # ---------------------------------------------------------------- 目录
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    models_dir: Path = PROJECT_ROOT / "outputs" / "models"
    checkpoints_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    logs_dir: Path = PROJECT_ROOT / "outputs" / "logs"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "reports"
    pretrained_dir: Path = field(
        default_factory=lambda: Path(
            _env_str("NEWSCLF_PRETRAINED_DIR", str(PROJECT_ROOT / "pretrained"))
        )
    )

    # ---------------------------------------------------------------- 数据
    max_len: int = field(default_factory=lambda: _env_int("NEWSCLF_MAX_LEN", 32))
    batch_size: int = field(default_factory=lambda: _env_int("NEWSCLF_BATCH_SIZE", 64))

    # ---------------------------------------------------------------- 随机森林
    rf_n_estimators: int = field(
        default_factory=lambda: _env_int("NEWSCLF_RF_TREES", 100)
    )
    rf_test_size: float = field(
        default_factory=lambda: _env_float("NEWSCLF_RF_TEST_SIZE", 0.2)
    )

    # ---------------------------------------------------------------- FastText
    fasttext_autotune: bool = field(
        default_factory=lambda: _env_bool("NEWSCLF_FASTTEXT_AUTOTUNE", True)
    )
    fasttext_autotune_duration: int = field(
        default_factory=lambda: _env_int("NEWSCLF_FASTTEXT_AUTOTUNE_SECONDS", 300)
    )
    fasttext_lr: float = field(default_factory=lambda: _env_float("NEWSCLF_FASTTEXT_LR", 0.1))
    fasttext_epoch: int = field(default_factory=lambda: _env_int("NEWSCLF_FASTTEXT_EPOCH", 25))
    fasttext_word_ngrams: int = field(
        default_factory=lambda: _env_int("NEWSCLF_FASTTEXT_WORD_NGRAMS", 2)
    )

    # ---------------------------------------------------------------- BERT
    bert_model_name: str = field(
        default_factory=lambda: _env_str("NEWSCLF_BERT_MODEL", "bert-base-chinese")
    )
    bert_epochs: int = field(default_factory=lambda: _env_int("NEWSCLF_BERT_EPOCHS", 3))
    bert_lr: float = field(default_factory=lambda: _env_float("NEWSCLF_BERT_LR", 5e-5))

    # ---------------------------------------------------------------- BiLSTM 蒸馏
    lstm_embed_size: int = 128
    lstm_num_layers: int = 3
    lstm_hidden_size: int = 256
    lstm_dropout: float = 0.3
    lstm_lr: float = field(default_factory=lambda: _env_float("NEWSCLF_LSTM_LR", 1e-3))
    distill_alpha: float = field(
        default_factory=lambda: _env_float("NEWSCLF_DISTILL_ALPHA", 0.7)
    )
    distill_temperature: float = field(
        default_factory=lambda: _env_float("NEWSCLF_DISTILL_TEMPERATURE", 2.0)
    )

    # ---------------------------------------------------------------- LLM
    llm_provider: str = field(
        default_factory=lambda: _env_str("NEWSCLF_LLM_PROVIDER", "deepseek").lower()
    )
    llm_timeout: float = field(default_factory=lambda: _env_float("NEWSCLF_LLM_TIMEOUT", 60.0))

    # ---------------------------------------------------------------- 其它
    prune_amount: float = field(
        default_factory=lambda: _env_float("NEWSCLF_PRUNE_AMOUNT", 0.3)
    )

    # ================================================================ 路径
    @property
    def train_file(self) -> Path:
        return self.raw_dir / "train.txt"

    @property
    def dev_file(self) -> Path:
        return self.raw_dir / "dev.txt"

    @property
    def test_file(self) -> Path:
        return self.raw_dir / "test.txt"

    @property
    def class_file(self) -> Path:
        return self.raw_dir / "class.txt"

    @property
    def stopwords_file(self) -> Path:
        return self.raw_dir / "stopwords.txt"

    @property
    def bert_pretrained_dir(self) -> Path:
        """本地预训练权重目录（不随仓库分发，需通过脚本下载）。"""
        return self.pretrained_dir / self.bert_model_name

    # ================================================================ 模型产物
    @property
    def tfidf_path(self) -> Path:
        return self.models_dir / "tfidf_vectorizer.joblib"

    @property
    def rf_path(self) -> Path:
        return self.models_dir / "random_forest.joblib"

    def fasttext_path(self, granularity: str = "word") -> Path:
        return self.models_dir / f"fasttext_{granularity}.bin"

    @property
    def bert_path(self) -> Path:
        return self.models_dir / "bert.pt"

    @property
    def bilstm_path(self) -> Path:
        return self.models_dir / "bilstm_student.pt"

    @property
    def bert_quantized_path(self) -> Path:
        return self.models_dir / "bert_quantized.pt"

    @property
    def bert_pruned_path(self) -> Path:
        return self.models_dir / "bert_pruned.pt"

    # ================================================================ 元信息
    @property
    def labels(self) -> Tuple[str, ...]:
        return read_labels(self.class_file)

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def device(self) -> str:
        """优先使用环境变量指定，否则自动探测 cuda / mps / cpu。"""
        forced = os.getenv("NEWSCLF_DEVICE")
        if forced:
            return forced
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def ensure_dirs(self) -> None:
        for path in (
            self.processed_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.reports_dir,
            self.pretrained_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # ================================================================ 校验
    def validate_data(self) -> None:
        missing = [
            str(p)
            for p in (self.train_file, self.dev_file, self.test_file, self.class_file)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "缺少数据文件：\n  - " + "\n  - ".join(missing) + "\n"
                "请确认 data/raw/ 下包含 train.txt / dev.txt / test.txt / class.txt。"
            )

    def as_dict(self) -> Dict[str, object]:
        """可公开序列化的配置快照。

        刻意不包含 ``root``：它是本机绝对路径，一旦写进 ``outputs/reports/*.json``
        并提交到仓库，就会泄露运行者的目录结构。项目根本就由代码推导，
        复现实验不需要它。
        """
        return {
            "seed": self.seed,
            "device": self.device,
            "labels": list(self.labels),
            "max_len": self.max_len,
            "batch_size": self.batch_size,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例，避免各处重复构造与重复读取文件。"""
    return Settings()


if __name__ == "__main__":  # pragma: no cover
    s = get_settings()
    print("项目根目录:", s.root)
    print("类别:", s.labels)
    print("设备:", s.device)
    print("模型目录:", s.models_dir)
