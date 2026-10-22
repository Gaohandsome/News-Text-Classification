# News Text Classification

> 中文新闻文本分类：从 TF-IDF / 随机森林、FastText、BERT，到 LLM 零样本与模型压缩的一站式实践。

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)

## 项目简介

给定一条中文新闻短文本，判断它属于 **10 个类别**中的哪一个：
`finance / realty / stocks / education / science / society / politics / sports / game / entertainment`。

项目覆盖一条完整的模型演进路线，**7 种方案共用同一套数据管线、配置和推理接口**：

| # | 方案 | 类型 | 一句话说明 |
| --- | --- | --- | --- |
| 1 | TF-IDF + 随机森林 | 传统机器学习 | 强基线，可解释性好，训练 5 分钟 |
| 2 | FastText | 浅层神经网络 | 词 + bigram 特征，CPU 上 0.3 ms 推理 |
| 3 | BERT 微调 | 预训练语言模型 | `bert-base-chinese` 全参数微调 |
| 4 | LLM 零样本 | 大语言模型 | DeepSeek / GLM / 通义千问，提示词约束输出 |
| 5 | 动态量化 | 模型压缩 | `nn.Linear` int8 量化，CPU 推理 |
| 6 | 知识蒸馏 | 模型压缩 | BERT 教师 → BiLSTM 学生 |
| 7 | 权重剪枝 | 模型压缩 | 注意力 query 全局非结构化 L1 剪枝 |

所有模型共享一个 `Predictor` 协议，因此 **网页端可以下拉切换模型**，命令行可以用
`--model` 指定，评估脚本可以用同一套流程给任意模型打分。

## 实验结果

完整环境、复现步骤与误差分析见 [docs/results.md](docs/results.md)。

| 模型 | 测试集 Acc | Macro F1 | 训练耗时 | 模型体积 | 单条推理 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + 随机森林 | 0.8294 | 0.8304 | 4 分 44 秒 | 1.5 GB | 16.2 ms |
| **FastText** | **0.9135** | **0.9135** | 27 秒 | 811 MB | **0.3 ms** |
| BERT 微调 | 待补测 | — | — | — | — |
| BERT + 量化 / 剪枝 | 待补测 | — | — | — | — |
| BiLSTM 蒸馏 | 待补测 | — | — | — | — |
| LLM 零样本 | 待补测 | — | — | — | — |

> 表中只填写了**本机实际跑出**的数字；BERT 系列需要预训练权重与 GPU，未实测就不编数字。
> 跑完后执行 `news-clf evaluate` 会自动生成 `outputs/reports/*.json`。

## 项目结构

```text
News-Text-Classification/
├── data/
│   ├── raw/                     # 原始语料（随仓库分发，11 MB）
│   └── processed/               # 分词后的中间数据（gitignore）
├── docs/
│   ├── architecture.md          # 分层设计、核心抽象、与原项目的对照
│   └── results.md               # 实验结果与复现说明
├── scripts/
│   └── download_pretrained.py   # 下载 bert-base-chinese（ModelScope / HF）
├── src/news_clf/
│   ├── config.py                # 唯一配置入口（路径自动推导 + 环境变量覆盖）
│   ├── labels.py                # 标签来源、中文释义、LLM 输出归一化
│   ├── data/                    # 读取 / 分词 / Dataset
│   ├── models/                  # BERT、BiLSTM、LLM 网络与封装
│   ├── training/                # 随机森林 / FastText / BERT / 蒸馏 / 量化 / 剪枝
│   ├── inference/               # 统一 Predictor 的 7 种实现
│   ├── registry.py              # 模型注册表（可用性检查 + 延迟加载）
│   ├── evaluate.py              # 任意模型 × 任意划分的统一评估
│   ├── serve/                   # FastAPI 应用 + 单页前端
│   └── cli.py                   # news-clf 命令行入口
├── tests/                       # 17 个单元测试
├── outputs/                     # 模型 / 日志 / 评估报告（权重不入库）
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## 快速开始

### 1. 安装

```bash
git clone <your-repo-url>
cd News-Text-Classification

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # 注册 news-clf 命令（也可用 python -m news_clf）
```

> Windows 上 `fasttext` 官方包需要本地编译，`requirements.txt` 已改用预编译的
> `fasttext-wheel`；GPU 版 PyTorch 请按 [pytorch.org](https://pytorch.org) 单独安装。

### 2. 数据

`data/raw/` 已包含全部数据（180,000 / 10,000 / 10,000 条，10 类完全均衡），无需额外下载。
先看一眼分布：

```bash
news-clf eda
```

### 3. 训练与压缩

```bash
news-clf prepare --target all              # 生成分词 / FastText 格式数据

news-clf train --model tfidf-rf            # 随机森林
news-clf train --model fasttext            # FastText（默认自动调参 300s）
news-clf train --model fasttext --no-autotune   # 想要更快时

python scripts/download_pretrained.py      # 拉取 bert-base-chinese（约 400 MB）
news-clf train --model bert                # BERT 微调
news-clf distill                           # BERT → BiLSTM 蒸馏
news-clf compress --method quantize        # int8 动态量化
news-clf compress --method prune --amount 0.3
```

### 4. 评估与预测

```bash
news-clf evaluate --model fasttext --split test
news-clf predict --model fasttext --text "沪指今日收涨1.2%，白酒板块领涨。"
news-clf models                            # 查看所有模型是否已就绪
```

### 5. 启动 Web 服务

```bash
news-clf serve --model fasttext --port 8000
```

打开 <http://127.0.0.1:8000>：左侧输入新闻、右侧显示类别与 Top-5 概率条，
顶部可切换模型（未训练的模型置灰）；接口文档在 <http://127.0.0.1:8000/docs>。

## 命令行一览

| 命令 | 作用 |
| --- | --- |
| `news-clf eda` | 类别分布、文本长度分布 |
| `news-clf prepare --target all\|rf\|fasttext` | 生成预处理数据（`--granularity word\|char`） |
| `news-clf train --model tfidf-rf\|fasttext\|bert` | 训练模型（`--epochs`、`--lr`、`--limit-batches`） |
| `news-clf distill` | 知识蒸馏（`--hard-from-gold` 可用人工标签做硬目标） |
| `news-clf compress --method quantize\|prune` | 模型压缩 |
| `news-clf evaluate --model <key> --split test` | 统一评估（`--limit` 适合 LLM 小样本试跑） |
| `news-clf predict --model <key> --text "..."` | 单条预测（也支持 `--file` 读文件） |
| `news-clf serve` | 启动 FastAPI 服务 |
| `news-clf models` | 列出模型及就绪状态 |

## 配置

路径与超参数集中在 `src/news_clf/config.py`，全部可用环境变量覆盖。
复制 `.env.example` 为 `.env` 后按需填写：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `NEWSCLF_LLM_PROVIDER` | `deepseek` | `deepseek` / `zhipu` / `dashscope` |
| `DEEPSEEK_API_KEY` / `ZHIPU_API_KEY` / `DASHSCOPE_API_KEY` | 空 | 用哪个供应商就填哪个 |
| `NEWSCLF_DEVICE` | 自动 | `cuda` / `mps` / `cpu` |
| `NEWSCLF_BATCH_SIZE` | `64` | 训练批大小 |
| `NEWSCLF_MAX_LEN` | `32` | 截断长度（数据 P95 = 25） |
| `NEWSCLF_BERT_EPOCHS` | `3` | BERT 训练轮次 |
| `NEWSCLF_FASTTEXT_AUTOTUNE` | `true` | 是否自动调参 |
| `NEWSCLF_PRUNE_AMOUNT` | `0.3` | 剪枝比例 |

> **安全提示**：API Key 只从环境变量读取，`.env` 已被 `.gitignore` 忽略。
> 请不要把真实 Key 写进源码或提交到仓库。

## 数据集

- 10 个类别，每类训练 18,000 条、验证 1,000 条、测试 1,000 条，**完全均衡**；
- 文本为新闻标题级短文本，平均 19.2 字，最长 38 字；
- 格式：`文本<TAB>类别ID`，类别 ID 与 `data/raw/class.txt` 的行号一致；
- 语料来自公开中文新闻集合，仅用于学习与研究，商用请自行确认授权。

## 设计说明

这是一个从"按课时拆分的 8 个脚本目录"重构而来的版本，主要改动：

| 问题（重构前） | 现状 |
| --- | --- |
| 8 份 `Config`，路径写死 `F:\...` | 1 份 `Settings`，路径由项目根自动推导，任意盘符可运行 |
| `dataloader / model / eval` 各复制 4 份 | 各 1 份，放在 `data/` 与 `models/` |
| 8 个 600+ 行的 Flask 脚本（约 5000 行重复） | 1 个 FastAPI 应用 + 模型注册表，页面内切换 |
| 7 个 `predict_fun` 各自为政 | 统一 `Predictor` 协议，预测逻辑只写一次 |
| 训练脚本靠 `python _05_xxx.py` 手动跑 | `news-clf` 子命令，参数可配 |
| `batch_x.to(device)` 未接收返回值，数据没真正上 GPU | 使用 `BatchEncoding.to(device)`，并由 `Config.device` 统一管理 |
| DeepSeek API Key 硬编码在源码 | 环境变量 + `.env.example`，`.env` 已忽略 |
| 无 README / 依赖声明 / 忽略规则 | README、`requirements.txt`、`pyproject.toml`、`.gitignore`、单元测试 |

更多设计细节见 [docs/architecture.md](docs/architecture.md)。

## 常见问题

**`ModuleNotFoundError: No module named 'news_clf'`**
没有安装包。执行 `pip install -e .`，或临时用 `PYTHONPATH=src python -m news_clf ...`。

**`news-clf` 不是可用的命令**
同样需要 `pip install -e .`；或直接用 `python -m news_clf`。

**模型文件在哪？为什么仓库里没有？**
权重体积大（随机森林 1.5 GB、FastText 811 MB、BERT 约 400 MB），已在 `.gitignore` 中排除，
按上文命令训练即可生成到 `outputs/models/`。

**没有 GPU 能跑吗？**
随机森林、FastText、BERT 推理都能在 CPU 跑；BERT 全量微调在 CPU 上会非常慢，建议 GPU。

**`import fasttext` 失败**
Windows 请安装 `fasttext-wheel`（已在 `requirements.txt` 中）；Linux/macOS 可用官方 `fasttext`。

**LLM 提示缺少 API Key**
在 `.env` 中填写对应供应商的 Key，并确认 `NEWSCLF_LLM_PROVIDER` 与之一致。

**BERT 权重下载失败**
手动下载 `bert-base-chinese`（`config.json`、`vocab.txt`、`tokenizer*.json`、权重），
放到 `pretrained/bert-base-chinese/` 即可。

## 测试

```bash
pip install -e ".[dev]"
pytest -q          # 17 passed
```

## License

[MIT](LICENSE) © 2026 gaoyu
