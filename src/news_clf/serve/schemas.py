"""HTTP 接口的数据模型。"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待分类的新闻正文")
    model: Optional[str] = Field(None, description="模型 key，缺省用服务默认模型")
    top_k: int = Field(5, ge=1, le=20, description="返回概率最高的前 k 个类别")


class ModelInfo(BaseModel):
    key: str
    display_name: str
    family: str
    description: str
    available: bool
    status: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]
    model: str
    latency_ms: float
