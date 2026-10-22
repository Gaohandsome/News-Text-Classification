"""BERT 分类模型。

与原始实现的区别：
- 预训练权重目录不再写死，由 ``Settings.bert_pretrained_dir`` 推导；
- 模型结构单独抽出 ``BertClassifier``，训练与推理共用同一份定义；
- ``forward`` 直接接收 ``input_ids / attention_mask / token_type_ids``，
  调用侧可以写 ``model(**batch)``，避免手工拼字典。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from news_clf.config import Settings, get_settings


def pretrained_dir(settings: Optional[Settings] = None) -> Path:
    """返回本地预训练模型目录，缺失时给出可执行的修复提示。"""
    settings = settings or get_settings()
    path = settings.bert_pretrained_dir
    if not path.exists() or not (path / "config.json").exists():
        raise FileNotFoundError(
            f"未找到 BERT 预训练模型: {path}\n"
            "请先执行：\n"
            "  python scripts/download_pretrained.py\n"
            "或手动下载 bert-base-chinese 后放到上述目录。"
        )
    return path


def load_tokenizer(settings: Optional[Settings] = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(pretrained_dir(settings)))


def load_bert_config(settings: Optional[Settings] = None):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(pretrained_dir(settings)))


class BertClassifier(torch.nn.Module):
    """BERT 编码器 + 线性分类头。"""

    def __init__(self, config, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **_ignored,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # BERT 的 pooler_output 即 [CLS] 经过一层全连接后的句向量
        return self.classifier(self.dropout(outputs.pooler_output))


def build_classifier(
    settings: Optional[Settings] = None, pretrained: bool = True
) -> BertClassifier:
    """构建分类模型。

    :param pretrained: True 时加载预训练权重（训练用），False 时只用结构（推理加载 checkpoint 用）
    """
    settings = settings or get_settings()
    config = load_bert_config(settings)
    model = BertClassifier(config, settings.num_labels)
    if pretrained:
        from transformers import BertModel

        model.bert = BertModel.from_pretrained(str(pretrained_dir(settings)))
    return model
