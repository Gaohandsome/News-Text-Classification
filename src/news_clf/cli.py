"""命令行入口。

用法示例::

    news-clf eda
    news-clf prepare --target all
    news-clf train --model tfidf-rf
    news-clf train --model bert --epochs 3
    news-clf distill --epochs 3
    news-clf compress --method quantize
    news-clf evaluate --model bert --split test
    news-clf predict --model bert --text "沪指今日收涨1.2%"
    news-clf serve --model bert --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from news_clf import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-clf",
        description="中文新闻文本分类：训练 / 压缩 / 评估 / 推理 / Web 服务",
    )
    parser.add_argument("--version", action="version", version=f"news-clf {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("eda", help="查看类别分布与文本长度分布")

    p_prepare = sub.add_parser("prepare", help="生成预处理数据")
    p_prepare.add_argument(
        "--target", choices=["all", "rf", "fasttext"], default="all"
    )
    p_prepare.add_argument("--granularity", choices=["word", "char"], default="word")

    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument(
        "--model",
        required=True,
        choices=["tfidf-rf", "rf", "fasttext", "bert"],
        help="tfidf-rf / fasttext / bert",
    )
    p_train.add_argument("--epochs", type=int, help="训练轮次（深度学习模型）")
    p_train.add_argument("--lr", type=float, help="学习率（深度学习模型）")
    p_train.add_argument("--granularity", choices=["word", "char"], default="word")
    p_train.add_argument(
        "--no-autotune", action="store_true", help="FastText 关闭自动调参，改用固定超参"
    )
    p_train.add_argument(
        "--limit-batches", type=int, help="每轮只跑前 N 个 batch，用于冒烟测试"
    )

    p_distill = sub.add_parser("distill", help="BERT -> BiLSTM 知识蒸馏")
    p_distill.add_argument("--epochs", type=int)
    p_distill.add_argument("--eval-steps", type=int, default=200)
    p_distill.add_argument(
        "--hard-from-gold", action="store_true", help="硬标签使用人工标注而非教师预测"
    )

    p_compress = sub.add_parser("compress", help="模型压缩")
    p_compress.add_argument("--method", choices=["quantize", "prune"], required=True)
    p_compress.add_argument("--amount", type=float, help="剪枝比例，默认 0.3")

    p_eval = sub.add_parser("evaluate", help="在数据集上评估模型")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--split", choices=["dev", "test"], default="test")
    p_eval.add_argument("--limit", type=int, help="只评估前 N 条（LLM 调试用）")

    p_predict = sub.add_parser("predict", help="单条文本预测")
    p_predict.add_argument("--model", help="模型 key，缺省取第一个可用模型")
    p_predict.add_argument("--text", help="待分类文本")
    p_predict.add_argument("--file", type=Path, help="从文件读取文本")
    p_predict.add_argument("--top-k", type=int, default=3)

    p_serve = sub.add_parser("serve", help="启动 Web 服务")
    p_serve.add_argument("--model", help="默认模型 key")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("models", help="列出所有模型及其可用状态")
    return parser


# --------------------------------------------------------------------- 子命令
def cmd_eda(_args: argparse.Namespace) -> int:
    from news_clf.data.eda import run_eda

    run_eda()
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from news_clf.data.preprocess import prepare_all, prepare_fasttext, prepare_rf

    if args.target == "rf":
        outputs = prepare_rf()
    elif args.target == "fasttext":
        outputs = prepare_fasttext(granularity=args.granularity)
    else:
        outputs = prepare_all(granularity=args.granularity)

    for name, result in outputs.items():
        print(f"[{name}]")
        for split, path in result.items():
            print(f"  {split}: {path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    model = args.model.lower()
    metrics: dict

    if model in {"tfidf-rf", "rf"}:
        from news_clf.training import rf

        metrics = rf.train()
    elif model == "fasttext":
        from news_clf.training import fasttext

        metrics = fasttext.train(
            granularity=args.granularity,
            autotune=False if args.no_autotune else None,
        )
    elif model == "bert":
        from news_clf.training import bert

        metrics = bert.train(
            epochs=args.epochs, lr=args.lr, limit_batches=args.limit_batches
        )
    else:  # pragma: no cover - argparse 已限制取值
        raise ValueError(model)

    print("\n训练完成，测试集指标:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    from news_clf.training import distillation

    metrics = distillation.train(
        epochs=args.epochs,
        eval_steps=args.eval_steps,
        hard_from_gold=args.hard_from_gold,
    )
    print("\n蒸馏完成，测试集指标:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    from news_clf.training import compression

    if args.method == "quantize":
        metrics = compression.quantize()
    else:
        metrics = compression.prune(amount=args.amount)

    print(f"\n{args.method} 完成:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from news_clf.evaluate import evaluate

    metrics = evaluate(args.model, split=args.split, limit=args.limit)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from news_clf.registry import create_predictor, first_available

    text = args.text
    if args.file:
        text = args.file.read_text(encoding="utf-8").strip()
    if not text:
        text = input("请输入新闻文本: ").strip()

    model = args.model or first_available()
    if not model:
        print("当前没有可用模型，请先训练。", file=sys.stderr)
        return 2

    predictor = create_predictor(model)
    prediction = predictor.predict(text)
    print(json.dumps(prediction.to_dict(top_k=args.top_k), ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from news_clf.serve.app import run

    run(host=args.host, port=args.port, default_model=args.model)
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    import unicodedata

    from news_clf.registry import list_models

    def width(text: str) -> int:
        """中文按 2 个字符宽度计算，保证表格对齐。"""
        return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)

    def pad(text: str, target: int) -> str:
        return text + " " * max(target - width(text), 0)

    models = list_models()
    key_width = max(width(model["key"]) for model in models)
    name_width = max(width(model["display_name"]) for model in models)

    header = f"{pad('KEY', key_width)}  {pad('模型', name_width)}  状态"
    print(header)
    print("-" * (width(header) + 20))
    for model in models:
        status = "就绪" if model["available"] else model["status"]
        print(
            f"{pad(model['key'], key_width)}  "
            f"{pad(model['display_name'], name_width)}  {status}"
        )
    return 0


HANDLERS = {
    "eda": cmd_eda,
    "prepare": cmd_prepare,
    "train": cmd_train,
    "distill": cmd_distill,
    "compress": cmd_compress,
    "evaluate": cmd_evaluate,
    "predict": cmd_predict,
    "serve": cmd_serve,
    "models": cmd_models,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = HANDLERS[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
