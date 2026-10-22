"""FastAPI 服务。

原始项目为 8 个模型各写了一份 600+ 行的 Flask 脚本，彼此只差 import 和端口；
这里合并为一个应用，通过请求参数（或页面上切换）选择模型。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from news_clf import __version__
from news_clf.config import Settings, get_settings
from news_clf.registry import (
    MODEL_SPECS,
    create_predictor,
    first_available,
    list_models,
    resolve_key,
)
from news_clf.serve.schemas import PredictRequest

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).with_name("templates") / "index.html"


def create_app(
    default_model: Optional[str] = None, settings: Optional[Settings] = None
) -> FastAPI:
    settings = settings or get_settings()
    default_model = (
        default_model
        or os.getenv("NEWSCLF_DEFAULT_MODEL")
        or first_available(settings)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """启动时预热默认模型，避免首个请求承担模型加载与 jieba 词典初始化的耗时。"""
        key = app.state.default_model
        if key and MODEL_SPECS.get(key) and key != "llm":
            try:
                predictor = create_predictor(key, settings)
                predictor.predict("预热文本")
                logger.info("默认模型已预热: %s", key)
            except Exception as exc:  # 预热失败不应阻止服务启动
                logger.warning("默认模型 %s 预热失败: %s", key, exc)
        yield

    app = FastAPI(
        title="News Text Classification API",
        version=__version__,
        description="中文新闻文本分类：传统机器学习 / FastText / BERT / LLM / 模型压缩",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.default_model = default_model

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        if not TEMPLATE_PATH.exists():  # pragma: no cover
            return HTMLResponse("<h1>页面模板缺失</h1>", status_code=500)
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))

    @app.get("/api/health", tags=["system"])
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "device": settings.device,
            "default_model": app.state.default_model,
            "num_labels": settings.num_labels,
        }

    @app.get("/api/models", tags=["system"])
    def models() -> dict:
        return {
            "default_model": app.state.default_model,
            "models": list_models(settings),
        }

    @app.get("/api/labels", tags=["system"])
    def labels() -> dict:
        from news_clf.labels import describe

        return {
            "labels": [
                {"name": label, "description": describe(label)}
                for label in settings.labels
            ]
        }

    @app.post("/api/predict", tags=["inference"])
    def predict(request: PredictRequest) -> dict:
        requested = request.model or app.state.default_model
        if not requested:
            raise HTTPException(
                status_code=409,
                detail="当前没有可用模型：请先训练模型，或配置 LLM 的 API Key。",
            )

        try:
            key = resolve_key(requested)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            predictor = create_predictor(key, settings)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # 例如缺少 API Key
            raise HTTPException(status_code=500, detail=f"模型加载失败: {exc}") from exc

        try:
            prediction = predictor.predict(request.text)
        except Exception as exc:
            logger.exception("预测失败")
            raise HTTPException(status_code=500, detail=f"预测失败: {exc}") from exc

        return prediction.to_dict(top_k=request.top_k)

    return app


# uvicorn news_clf.serve.app:app
app = create_app()


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    default_model: Optional[str] = None,
) -> None:
    import uvicorn

    model = default_model or first_available()
    print("=" * 62)
    print("  新闻文本分类服务")
    print(f"  页面: http://{host}:{port}")
    print(f"  文档: http://{host}:{port}/docs")
    print(f"  默认模型: {model or '无（请先训练模型）'}")
    print("=" * 62)
    uvicorn.run(create_app(default_model), host=host, port=port)
