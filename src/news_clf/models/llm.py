"""LLM 零样本 / 少样本分类器。

统一走 OpenAI 兼容协议，因此只需切换 base_url + 模型名即可在
DeepSeek / 智谱 GLM / 阿里云百炼 之间切换。

API Key 一律从环境变量读取（原项目把 DeepSeek Key 硬编码在源码里，
一旦推到 GitHub 会立即泄露），变量名见 ``PROVIDERS``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence, Tuple

from news_clf.config import Settings, get_settings
from news_clf.labels import canonical_label, describe


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    default_model: str


PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        name="deepseek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
    "zhipu": Provider(
        name="zhipu",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        default_model="glm-4-flash",
    ),
    "dashscope": Provider(
        name="dashscope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen-plus",
    ),
}


def build_system_prompt(labels: Sequence[str]) -> str:
    """根据 class.txt 动态生成提示词，保证类别集合永远和标签文件一致。"""
    category_lines = "\n".join(f"- {label}：{describe(label)}" for label in labels)
    examples = "\n".join(
        f"文本：{_few_shot_text(label)}\n类别：{label}" for label in labels
    )
    return f"""# 角色
你是一个专业的中文新闻文本分类助手。

# 任务
阅读用户提供的新闻文本，从下列固定类别中选出唯一一个最匹配的类别。

# 类别定义（严格限定，不得新增或修改）
{category_lines}

# 分类规则
1. 只能输出上述类别中的一个英文单词，禁止输出解释、标点或换行。
2. 若文本涉及多个领域，选择其核心主题对应的类别。
3. 若难以判断，选择最接近的类别，不要拒答、不要输出"未知"。

# 易混淆类别
- finance vs stocks：宏观经济/银行/货币政策 → finance；个股/大盘/证券行情 → stocks
- science vs game：AI/科研本身 → science；游戏产品/电竞 → game
- society vs politics：民生事件本身 → society；政府政策/党政行为 → politics

# 少样本示例
{examples}
"""


def _few_shot_text(label: str) -> str:
    samples = {
        "finance": "央行今日宣布下调存款准备金率 0.5 个百分点。",
        "realty": "某小区业主集体维权，要求开发商整改房屋质量问题。",
        "stocks": "沪指今日收涨 1.2%，白酒板块领涨。",
        "education": "教育部发布新规，中小学课后服务将全面覆盖。",
        "science": "我国科学家在量子计算领域取得重大突破。",
        "society": "某市地铁早高峰发生故障，大量乘客滞留。",
        "politics": "外交部就相关问题作出回应，强调维护国家主权。",
        "sports": "国足在世预赛中 2:1 战胜对手，取得关键三分。",
        "game": "某热门手游新版本上线，首日流水破亿。",
        "entertainment": "某知名演员官宣结婚，相关话题登上热搜第一。",
    }
    return samples.get(label, f"一段关于{label}的新闻文本。")


class LLMClassifier:
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        provider = PROVIDERS.get(settings.llm_provider)
        if provider is None:
            raise ValueError(
                f"不支持的 LLM 供应商: {settings.llm_provider!r}，"
                f"可选 {sorted(PROVIDERS)}"
            )
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"未配置 {provider.api_key_env}，请在 .env 或环境变量中设置后再使用 LLM 分类。"
            )

        from openai import OpenAI

        self.provider = provider
        self.model = os.getenv("NEWSCLF_LLM_MODEL", provider.default_model)
        self.labels = settings.labels
        self.system_prompt = build_system_prompt(self.labels)
        self.client = OpenAI(
            api_key=api_key, base_url=provider.base_url, timeout=settings.llm_timeout
        )

    def classify(self, text: str) -> Tuple[str, str]:
        """返回 ``(规范化类别, 原始回复)``。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            stream=False,
        )
        raw = (response.choices[0].message.content or "").strip()
        label = canonical_label(raw, self.labels)
        if label is None:
            raise ValueError(f"LLM 返回了无法识别的类别: {raw!r}")
        return label, raw
