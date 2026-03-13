# TextDedup

TextDedup 是一个面向文本去重与疑似改写识别的两层检索原型仓库。

当前项目聚焦于以下处理链路：

- 第一层：基于指纹的快速去重与候选召回，当前已实现 `SimHash`，后续可扩展 `MinHash`
- 第二层：基于向量的语义匹配，用于对疑似相似文本做深度判别，目标模型为 `BERT` / `SBERT`

数据采集不再作为本仓库交付内容。后续数据将由外部流程预先准备，并通过配置方式接入本项目。

说明：当前代码中两阶段检索的精排部分仍是 `TF-IDF + cosine similarity` 原型，用于验证接口与流程，后续会替换为向量语义匹配实现。

## 快速开始

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 当前能力

- `SimHash` 指纹生成、汉明距离计算与粗粒度相似度估计
- `SimilarityEngine`：基于 `TF-IDF + cosine similarity` 的文本相似度原型
- `TwoStageSearchEngine`：先用 `SimHash` 过滤候选，再做二阶段精排

## 后续演进方向

- 增加可插拔的一层候选召回策略，支持 `SimHash` / `MinHash`
- 将二阶段精排从 `TF-IDF` 迁移到向量语义模型，如 `BERT` / `SBERT`
- 通过统一配置接入离线采集好的文本数据、片段数据或向量索引

## 目录说明

- `src/textdedup`：文本去重与两阶段检索核心实现
- `tests`：算法模块单元测试
- `docs`：技术设计与交接文档
- `data`：预留的数据与配置接入目录
