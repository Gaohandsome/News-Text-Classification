# 实验结果

## 1. 运行环境

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows 11 |
| Python | 3.12.7 |
| PyTorch | 2.6.0+cpu（**无 GPU**） |
| scikit-learn | 1.5.1 |
| 数据 | train 180,000 / dev 10,000 / test 10,000，10 类完全均衡 |
| 文本长度 | 平均 19.2 字，最大 38，P95 = 25（对应 `max_len=32`） |

## 2. 复现方式

```bash
# 1) 准备预处理数据（约 30 秒）
news-clf prepare --target all

# 2) 训练基线
news-clf train --model tfidf-rf          # 约 5 分钟（CPU）
news-clf train --model fasttext --no-autotune   # 约 30 秒（CPU）

# 3) 深度学习方案（需先下载预训练权重，建议 GPU）
python scripts/download_pretrained.py
news-clf train --model bert
news-clf distill
news-clf compress --method quantize
news-clf compress --method prune

# 4) 统一评估（任意模型 × 任意划分）
news-clf evaluate --model fasttext --split test
```

所有指标会写入 `outputs/reports/<model>_<split>_metrics.json`。

## 3. 结果

### 3.1 已实测（CPU）

| 模型 | 测试集 Acc | Macro Precision | Macro Recall | Macro F1 | 训练耗时 | 模型体积 | 单条推理 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + 随机森林 | 0.8294 | 0.8331 | 0.8294 | 0.8304 | 4 分 44 秒 | 1.5 GB | 16.2 ms |
| FastText（word + bigram） | **0.9135** | 0.9135 | 0.9135 | **0.9135** | 27 秒 | 811 MB | 0.3 ms |

补充：随机森林在训练集内部的 20% 留出集上为 Acc 0.8254 / F1 0.8266，
与原始项目记录的 "0.82 左右" 一致。

### 3.2 待补测（需预训练权重 / GPU）

| 模型 | 状态 | 说明 |
| --- | --- | --- |
| BERT 微调 | 待跑 | 需要 `bert-base-chinese` 权重；CPU 全量微调约需数小时 |
| BERT 动态量化 | 待跑 | 依赖 BERT 权重，量化后建议在 CPU 上测延迟 |
| BiLSTM 蒸馏 | 待跑 | 依赖 BERT 教师模型 |
| BERT 剪枝 | 待跑 | 依赖 BERT 权重 |
| LLM 零样本 | 待跑 | 需配置 `DEEPSEEK_API_KEY`，可用 `--limit 200` 先小样本试跑 |

> 这几项本人未在当前机器实测，因此不填写具体数字，避免出现无法复现的结果。

## 4. 结果分析

**为什么 FastText 明显优于随机森林？**

1. 数据是**标题级短文本**（平均 19 字），词共现信息有限，n-gram（`wordNgrams=2`）能补足短语特征；
2. TF-IDF + 随机森林需要把 11 万维稀疏向量喂给 100 棵树，每次分裂只能看部分维度，
   对短文本的判别信息利用不充分；
3. FastText 的 subword / n-gram 哈希把未登录词也映射到向量空间，泛化更稳。

**模型体积为什么这么大？**

- 随机森林：`112,420` 维 TF-IDF × 100 棵完全生长的树，叶子节点存储大量阈值，joblib 落盘 1.5 GB；
- FastText：默认 `dim=100`、`bucket=2,000,000`，仅哈希桶矩阵就有 2M × 100 × 4B ≈ 800 MB。

两者都属于"训练快、部署重"的方案。若要压缩体积，可：
- 随机森林：限制 `max_depth`、降低 `n_estimators`、对 TF-IDF 做 `max_features` 截断；
- FastText：减小 `bucket`（如 200,000）与 `dim`（如 50），精度损失通常在 1 个百分点内。

**延迟说明**

表中的单条推理为**预热后**的服务端耗时（`Predictor.predict` 内部计时，
不含网络往返）。首次调用会多出 0.6~1.1 秒，主要是 jieba 词典加载；
服务启动时已自动预热，因此线上首个请求不会承担这部分开销。

## 5. 复现注意事项

- 随机森林训练使用了 `random_state=42` 与分层切分，结果可复现；
- FastText 使用固定超参（`lr=0.1, epoch=25, wordNgrams=2`），未开启自动调参；
  开启 `--autotune` 会额外搜索 300 秒，通常能再提升 0.5~1 个百分点，但耗时不可控；
- 所有实验均在 CPU 上完成，GPU 环境数值可能因浮点归约顺序略有差异（±0.2%）。
