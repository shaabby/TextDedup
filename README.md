# TextDedup

TextDedup 是一个面向文本去重与改写识别的离线检索原型仓库。

当前主链路已经收敛为：

- 第一阶段：`TF-IDF + cosine` 候选重排
- 第二阶段：`SBERT` 语义精排（可开关）

另外提供可切换的 `sbert-only` 模式，用于绕过词特征召回限制，直接做向量语义检索。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 一键查询

使用默认配置：

```bash
python scripts/query_similar_segments.py --config configs/query_default.toml
```

默认脚本支持：

- 多目录扫描 + `glob` 过滤输入文件
- 文本切片（`paragraph` / `sliding` / `hybrid`）
- `two-stage`（TF-IDF + SBERT）与 `sbert-only` 切换
- 两类缓存：
  - `two-stage` 切片缓存
  - `sbert-only` 向量缓存
- 终端进度输出（可调频率）

## 推荐配置

项目内置三套预设：

- `configs/query_fast.toml`：速度优先
- `configs/query_recall.toml`：召回优先
- `configs/query_sbert_only.toml`：纯 SBERT + 缓存

示例：

```bash
python scripts/query_similar_segments.py --config configs/query_fast.toml
python scripts/query_similar_segments.py --config configs/query_recall.toml
python scripts/query_similar_segments.py --config configs/query_sbert_only.toml
```

## 关键参数

以 `two-stage` 为例：

- `pipeline.tfidf_candidate_k`：进入 TF-IDF 重排的候选数
- `pipeline.sbert_top_n`：进入 SBERT 的候选数
- `pipeline.enable_tfidf`：是否启用 TF-IDF（`two-stage` 必须为 `true`）
- `pipeline.enable_sbert`：是否启用 SBERT 终排

运行时：

- `runtime.progress_interval`：进度输出频率
- `runtime.quiet_progress`：是否关闭进度日志

## 缓存机制

### two-stage 缓存

- 内容：切片文本与元信息
- 作用：避免重复读取大文件与重复切片
- 配置：
  - `cache.chunk_text_cache_file`
  - `cache.chunk_text_cache_meta_file`
  - `cache.reuse_chunk_text_cache`

### sbert-only 缓存

- 内容：文档向量（embeddings）
- 作用：避免重复全量编码
- 配置：
  - `cache.sbert_embeddings_file`
  - `cache.sbert_meta_file`
  - `cache.reuse_sbert_cache`

两类缓存都使用签名校验（文件时间戳/大小 + 关键配置），命中才复用，变更则自动重建。

## 当前能力

- `SimilarityEngine`：`TF-IDF + cosine` 检索
- `SbertSimilarityEngine`：本地离线 SBERT 编码与检索
- `TwoStageSearchEngine`：`TF-IDF -> (optional) SBERT`
- `scripts/query_similar_segments.py`：离线查询主脚本

## 说明

- 项目仍保留 `simhash.py` 与 `cpp_bridge.py` 作为独立实验/兼容模块。
- 但在当前主查询链路中，已不再使用 SimHash 作为候选召回接口。
