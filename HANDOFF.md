# TextDedup Handoff

## 1. Repository Scope

Workspace root:
- `/home/user/code/TextDedup`

当前 Git 仓库只交付 `TextDedup` 主模块，不再包含任何爬取或站点采集子项目。

当前仓库定位：
- 第一层：基于文本指纹的快速去重与候选过滤
- 第二层：基于向量的语义匹配与改写识别
- 数据入口：外部预采集数据通过配置接入，不在仓库内负责采集

## 2. Implemented Modules

- `src/textdedup/similarity.py`
  - `SimilarityEngine`
  - `pairwise_similarity`
  - 当前为 `TF-IDF + cosine similarity` 原型实现
- `src/textdedup/simhash.py`
  - `SimHash` 指纹生成与汉明距离计算
- `src/textdedup/two_stage.py`
  - `TwoStageSearchEngine`
  - Stage 1: `SimHash` candidate filtering
  - Stage 2: `TF-IDF` reranking prototype
- `src/textdedup/__init__.py`
  - 导出 `SimilarityEngine`、`pairwise_similarity`、`SimHash`、`TwoStageResult`、`TwoStageSearchEngine`

说明：二阶段“向量语义匹配”是后续目标架构，当前代码尚未落地 `BERT` / `SBERT` 编码与向量索引。

## 3. Test Coverage

- `tests/test_similarity.py`
- `tests/test_simhash.py`
- `tests/test_two_stage.py`

运行：

```bash
/home/user/code/TextDedup/.venv/bin/python -m pytest -q
```

## 4. Recommended Data Access Pattern

后续数据流建议统一为：

1. 在仓库外完成原始数据采集与清洗。
2. 通过配置指定数据文件、字段映射和索引参数。
3. 在本仓库内部完成指纹召回、候选筛选和语义复核。

这样可以把“数据来源”与“去重算法”解耦，避免仓库职责再次膨胀。

## 5. Recommended Next Steps

1. 为第一层候选召回抽象统一接口，补上 `MinHash` 扩展点。
2. 将 `TwoStageSearchEngine` 的二阶段精排替换为 `BERT` / `SBERT` 向量相似度实现。
3. 增加配置驱动的数据加载层，支持直接接入已采集好的文本或片段数据。
