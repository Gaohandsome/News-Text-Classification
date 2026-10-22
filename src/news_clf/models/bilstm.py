"""知识蒸馏的学生模型：BiLSTM。

保留原始实现的全部细节：复用 BERT 词表做 Embedding、过滤 [CLS]/[SEP]/[PAD]、
对有效 token 做平均池化。额外给池化分母加了 clamp_min(1)，避免空句子除零。
"""

from __future__ import annotations

from typing import Optional

import torch

from news_clf.config import Settings, get_settings


class BiLSTMClassifier(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.bilstm = torch.nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size * 2, num_labels)
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_ignored,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedded = self.embedding(input_ids)

        # 去掉 BERT 的特殊符号，只保留真实字符
        keep_mask = (input_ids != self.cls_token_id) & (input_ids != self.sep_token_id)
        valid_mask = (keep_mask & attention_mask.bool()).unsqueeze(-1).to(embedded.dtype)
        valid_embed = embedded * valid_mask

        lstm_out, _ = self.bilstm(valid_embed)
        valid_lstm_out = lstm_out * valid_mask

        summed = valid_lstm_out.sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / counts

        return self.classifier(self.dropout(pooled))


def build_bilstm(settings: Optional[Settings] = None) -> BiLSTMClassifier:
    from news_clf.models.bert import load_tokenizer

    settings = settings or get_settings()
    tokenizer = load_tokenizer(settings)
    return BiLSTMClassifier(
        vocab_size=len(tokenizer),
        num_labels=settings.num_labels,
        embed_size=settings.lstm_embed_size,
        hidden_size=settings.lstm_hidden_size,
        num_layers=settings.lstm_num_layers,
        dropout=settings.lstm_dropout,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
    )
