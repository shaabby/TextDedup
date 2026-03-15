# TextDedup 技术文档

## 1. 目标与范围

TextDedup 聚焦离线文本检索与改写识别，不负责数据采集。

当前目标：

- 以可解释、可复现、可缓存的方式完成离线检索
- 在保证召回的同时控制算力成本
- 提供可配置的两种检索模式

## 2. 总体架构

### 2.1 two-stage（默认）

`TF-IDF -> (optional) SBERT`

- Stage 1（TF-IDF）
  - 在全量切片语料上建词特征空间
  - 用 `tfidf_candidate_k` 控制候选规模
- Stage 2（SBERT，可选）
  - 对 TF-IDF 候选做语义重排
  - 用 `sbert_top_n` 控制进入终排的候选数

### 2.2 sbert-only（可选）

- 直接对全量切片做向量检索
- 强依赖向量缓存，否则首次编码开销较大

## 3. 关键模块

- `similarity.py`
  - `SimilarityEngine`
  - 负责 TF-IDF fit/query
- `sbert_similarity.py`
  - `SbertSimilarityEngine`
  - 负责向量编码与相似度检索
- `two_stage.py`
  - `TwoStageSearchEngine`
  - 负责二阶段编排与打分输出
- `chunking.py`
  - 文本切片（paragraph/sliding/hybrid）
- `scripts/query_similar_segments.py`
  - 统一 CLI、配置解析、缓存调度、进度输出

## 4. 配置模型

核心字段：

- `[pipeline]`
  - `retrieval_mode`
  - `enable_tfidf`
  - `enable_sbert`
  - `tfidf_candidate_k`
  - `sbert_top_n`
- `[chunking]`
  - `mode`
  - `window_sentences`
  - `stride_sentences`
- `[cache]`
  - `reuse_chunk_text_cache`
  - `reuse_sbert_cache`

约束：

- `two-stage` 模式必须启用 TF-IDF
- `sbert-only` 模式下强制启用 SBERT

## 5. 缓存设计

### 5.1 two-stage 缓存

- 缓存对象：切片文本与元信息
- 目标：跳过重复的读文件与切片过程

### 5.2 sbert-only 缓存

- 缓存对象：embeddings
- 目标：跳过重复全量编码

### 5.3 失效策略

签名包含：

- 输入文件路径/大小/mtime
- 文本字段映射
- 切片参数
- 模型关键配置

签名不一致则自动重建。

## 6. 性能与调参

### 6.1 默认参数（当前语料）

- `chunk_mode=sliding`
- `window_sentences=6`
- `stride_sentences=3`
- `tfidf_candidate_k=800`
- `sbert_top_n=20`

### 6.2 调参建议

- 速度优先：降低 `tfidf_candidate_k` / 关闭 `enable_sbert`
- 召回优先：增大 `tfidf_candidate_k` / 增大 `sbert_top_n`
- 改写较重：优先 `sbert-only` + 缓存

## 7. 已弃用说明

- SimHash 已从主查询链路移除，不再作为候选接口。
- 仓库中的 `simhash.py` 与 `cpp_bridge.py` 当前仅作为独立实验能力保留。

## 8. 后续演进

1. 引入批量评测入口与指标汇总（Recall@K、MRR、耗时）。
2. 支持分语料模板自动选参。
3. 引入向量索引（如 FAISS）以降低 sbert-only 大库检索成本。
