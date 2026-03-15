# TextDedup Handoff

## 1. 仓库定位

- 路径：`/home/user/code/TextDedup`
- 当前定位：离线文本检索与改写识别
- 主链路：`TF-IDF -> (optional) SBERT`
- 可选模式：`sbert-only`

说明：主查询脚本已移除 SimHash 依赖，不再把 SimHash 作为线上/主流程候选接口。

## 2. 核心模块

- `src/textdedup/similarity.py`
  - `SimilarityEngine`
  - 词特征检索（TF-IDF + cosine）
- `src/textdedup/sbert_similarity.py`
  - `SbertSimilarityEngine`
  - 本地离线模型语义检索
- `src/textdedup/two_stage.py`
  - `TwoStageSearchEngine`
  - 当前为二阶段：`TF-IDF -> (optional) SBERT`
- `scripts/query_similar_segments.py`
  - 统一查询入口（目录扫描、切片、缓存、进度输出）

## 3. 配置与预设

常用配置：

- `configs/query_default.toml`
- `configs/query_fast.toml`
- `configs/query_recall.toml`
- `configs/query_sbert_only.toml`

关键开关：

- `pipeline.retrieval_mode`: `two-stage | sbert-only`
- `pipeline.enable_tfidf`
- `pipeline.enable_sbert`
- `pipeline.tfidf_candidate_k`
- `pipeline.sbert_top_n`

## 4. 缓存策略

- `two-stage`：缓存切片文本（降低重复建库成本）
- `sbert-only`：缓存 embeddings（降低重复编码成本）
- 两者都采用签名校验，配置或数据变化后自动失效重建

## 5. 测试覆盖

- `tests/test_similarity.py`
- `tests/test_sbert_similarity.py`
- `tests/test_two_stage.py`
- `tests/test_chunking.py`
- `tests/test_export_shard_artifact_report.py`

运行：

```bash
/home/user/code/TextDedup/.venv/bin/python -m pytest -q
```

## 6. 当前默认实践

- 切片：`sliding` + `window=6` + `stride=3`
- 主模式：`two-stage`
- 日常调参优先改：`tfidf_candidate_k`、`sbert_top_n`

## 7. 后续建议

1. 增加批量 query 入口（文件列表/stdin）用于回归评测。
2. 增加命中高亮与结果解释字段，提升人工复核效率。
3. 为不同语料类型沉淀默认参数模板（小说、新闻、通用网页）。
