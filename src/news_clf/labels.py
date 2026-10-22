"""类别标签工具。

标签的唯一来源是 ``data/raw/class.txt``（每行一个类别），
避免像原项目那样把类别名和类别数量散落在多个脚本里。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# 类别中文释义，用于 LLM 提示词与前端展示；缺失的类别会退化为类别名本身。
LABEL_DESCRIPTIONS: Dict[str, str] = {
    "finance": "金融、银行、货币政策、投资理财、宏观经济",
    "realty": "房地产、楼市、房价、土地、物业管理",
    "stocks": "股票、股市、证券、基金、上市公司行情",
    "education": "教育、学校、考试、升学、培训、教育政策",
    "science": "科学技术、科研发现、航天、生物、物理、AI 技术",
    "society": "社会民生、公共事件、医疗、交通、环境",
    "politics": "政治、政府、政策、外交、党政、选举",
    "sports": "体育竞技、赛事、运动员、球类、奥运",
    "game": "电子游戏、网游、手游、电竞、游戏产业",
    "entertainment": "娱乐、明星、影视、综艺、音乐、八卦",
}


@lru_cache(maxsize=8)
def read_labels(path: Path) -> Tuple[str, ...]:
    """读取类别文件，返回有序元组（顺序即标签 id）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"类别文件不存在: {path}")
    labels = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not labels:
        raise ValueError(f"类别文件为空: {path}")
    duplicates = {x for x in labels if labels.count(x) > 1}
    if duplicates:
        raise ValueError(f"类别文件存在重复标签: {sorted(duplicates)}")
    return labels


def label2id(labels: Sequence[str]) -> Dict[str, int]:
    return {label: index for index, label in enumerate(labels)}


def id2label(labels: Sequence[str]) -> Dict[int, str]:
    return {index: label for index, label in enumerate(labels)}


def describe(label: str) -> str:
    return LABEL_DESCRIPTIONS.get(label, label)


def canonical_label(raw: str, labels: Sequence[str]) -> Optional[str]:
    """把 LLM 的自由文本输出归一化到固定类别。

    依次尝试：完全匹配 -> 去掉标点后匹配 -> 子串包含 -> 失败返回 None。
    """
    if not raw:
        return None
    text = raw.strip().lower()
    text = re.sub(r"[\s`'\"“”‘’。，、：:！!？?\-—_*#]+", "", text)
    if not text:
        return None

    lowered = {label.lower(): label for label in labels}
    if text in lowered:
        return lowered[text]

    # 去掉前缀，例如 "类别：finance" / "label: sports"
    for label in labels:
        if text.endswith(label.lower()) or text.startswith(label.lower()):
            return label

    # 兜底：子串包含（取最长匹配，避免 game 命中 entertainment 之类）
    hits = [label for label in labels if label.lower() in text]
    if hits:
        return max(hits, key=len)
    return None
