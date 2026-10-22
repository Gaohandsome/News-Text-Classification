"""探索性数据分析（EDA）：类别均衡性与文本长度分布。"""

from __future__ import annotations

import logging
import statistics

from news_clf.config import Settings, get_settings
from news_clf.data.loader import SPLITS, class_counts, read_split

logger = logging.getLogger(__name__)


def describe_split(split: str, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    examples = read_split(split, settings)
    lengths = [len(example.text) for example in examples]
    counts = class_counts(examples, settings.labels)

    summary = {
        "split": split,
        "num_examples": len(examples),
        "num_labels": len(counts),
        "class_counts": counts,
        "length": {
            "mean": round(statistics.fmean(lengths), 2),
            "max": max(lengths),
            "min": min(lengths),
            "stdev": round(statistics.pstdev(lengths), 2),
            "p95": sorted(lengths)[int(len(lengths) * 0.95) - 1],
        },
    }
    return summary


def run_eda(settings: Settings | None = None) -> dict[str, dict]:
    """对 train / dev / test 三个划分依次做 EDA 并打印报告。"""
    settings = settings or get_settings()
    settings.validate_data()
    report: dict[str, dict] = {}

    for split in SPLITS:
        summary = describe_split(split, settings)
        report[split] = summary
        total = summary["num_examples"]
        print("=" * 66)
        print(f"[{split}] 样本数: {total}")
        print("-" * 66)
        for label, count in summary["class_counts"].items():
            ratio = count / total * 100
            bar = "█" * max(1, int(ratio / 2))
            print(f"  {label:<14}{count:>7}  {ratio:>5.2f}%  {bar}")
        length = summary["length"]
        print(
            f"  文本长度  平均 {length['mean']} / 最大 {length['max']} / "
            f"最小 {length['min']} / 标准差 {length['stdev']} / P95 {length['p95']}"
        )
    print("=" * 66)
    return report
