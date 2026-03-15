# Changelog

All notable changes to this project will be documented in this file.
本文件记录项目所有重要变更。

## [Unreleased] - 2026-03-14

### Added
- Added Chinese word-level tokenization via `jieba` as the shared preprocessing path across retrieval stages.
  新增基于 `jieba` 的中文词级分词，作为各检索阶段共享预处理路径。
- Added external stopword files:
  新增外部停用词文件：
  - `data/stopwords/zh_common.txt`
  - `data/stopwords/zh_domain.txt`
- Added user dictionary support for Chinese proper nouns:
  新增中文专名用户词典支持：
  - default dictionary `data/dicts/zh_user_dict.txt`
    默认词典：`data/dicts/zh_user_dict.txt`
  - title-candidate extraction script `scripts/build_title_user_dict.py`
    标题候选词提取脚本：`scripts/build_title_user_dict.py`
- Added SBERT stage implementation with offline local-model loading support:
  新增 SBERT 阶段实现，支持离线本地模型加载：
  - `src/textdedup/sbert_similarity.py`
  - `SBERTConfig.local_model_path`
  - local model directory `models/bge-small-zh-v1.5`
    本地模型目录：`models/bge-small-zh-v1.5`
- Added chunk-based retrieval tooling:
  新增基于片段的检索工具：
  - chunking module `src/textdedup/chunking.py`
    切片模块：`src/textdedup/chunking.py`
  - query CLI `scripts/query_similar_segments.py`
    查询命令行工具：`scripts/query_similar_segments.py`
  - chunking tests `tests/test_chunking.py`
    切片测试：`tests/test_chunking.py`

### Changed
- Refactored the main retrieval path from three-stage (`SimHash -> TF-IDF -> SBERT`) to two-stage (`TF-IDF -> SBERT`), while keeping `sbert-only` as a selectable mode.
  主检索链路由三阶段（`SimHash -> TF-IDF -> SBERT`）重构为二阶段（`TF-IDF -> SBERT`），并保留 `sbert-only` 可选模式。
- Removed SimHash-related CLI/config interfaces from the main query script and unified candidate control on `tfidf_candidate_k`.
  主查询脚本移除 SimHash 相关 CLI/配置接口，并统一由 `tfidf_candidate_k` 控制候选规模。
- Simplified two-stage cache payload to chunk-text cache (no SimHash index persistence in main flow).
  two-stage 缓存简化为切片文本缓存（主流程不再持久化 SimHash 索引）。
- Earlier in this unreleased cycle, the pipeline was upgraded from two-stage to three-stage retrieval (historical milestone):
  在本次未发布迭代的早期，检索流程曾由两阶段升级为三阶段（历史里程碑）：
  - Stage 1: `SimHash` candidate recall
    第一阶段：`SimHash` 候选召回
  - Stage 2: `TF-IDF` rerank
    第二阶段：`TF-IDF` 重排
  - Stage 3: `SBERT` semantic rerank (optional, local offline supported)
    第三阶段：`SBERT` 语义重排（可开关，支持本地离线）
- Switched TF-IDF to word analyzer with shared tokenizer behavior and aligned token rules with preprocessing.
  TF-IDF 切换为词级分析器，与共享分词器和预处理 token 规则对齐。
- Updated SimHash weighting to use corpus IDF with optional cap and proper-noun downweighting.
  SimHash 加权更新为语料 IDF，并支持 IDF 上限与专名降权。
- Moved stopword handling from code defaults to file-based loading for easier maintenance.
  停用词机制由代码内置默认值迁移为文件加载，便于维护。
- Updated artifact report generation to include SBERT-related metadata and three-stage preview results.
  Artifact 报告增加 SBERT 相关元数据与三阶段预览结果。
- Tuned default chunk retrieval strategy for current corpus:
  针对当前语料调优默认片段检索策略：
  - default mode `sliding`
    默认模式：`sliding`
  - default window/stride `6/3`
    默认窗口/步长：`6/3`
  - default `candidate_k=800`
    默认候选数：`candidate_k=800`

### Fixed
- Fixed SBERT local model precedence so `local_model_path` is used before remote model name.
  修复 SBERT 本地模型优先级：优先使用 `local_model_path`，再回退远程模型名。
- Fixed report-term weighting alignment so displayed TF-IDF/IDF values match effective runtime caps/weights.
  修复报告词项权重对齐，确保展示的 TF-IDF/IDF 与运行时有效权重一致。
- Fixed long-paragraph chunk offset handling in chunk construction.
  修复长段落切片时的偏移计算问题。

### Tests
- Updated `tests/test_two_stage.py` to validate the new `TF-IDF + optional SBERT` behavior and removed SimHash-coupled assertions.
  更新 `tests/test_two_stage.py` 以验证新的 `TF-IDF + 可选 SBERT` 行为，并移除与 SimHash 绑定的断言。
- Current full suite status after refactor: `66 passed`.
  重构后的全量测试状态：`66 passed`。
- Added and passed SBERT-focused tests in `tests/test_sbert_similarity.py`.
  新增并通过 SBERT 专项测试：`tests/test_sbert_similarity.py`。
- Added script-level coverage in `tests/test_export_shard_artifact_report.py`.
  新增脚本级覆盖：`tests/test_export_shard_artifact_report.py`。
- Added chunking behavior tests in `tests/test_chunking.py`.
  新增切片行为测试：`tests/test_chunking.py`。
- Intermediate full suite status during this unreleased cycle: `64 passed`.
  本次未发布迭代过程中的阶段性全量测试状态：`64 passed`。

### Documentation
- Fully refreshed architecture docs to match the current pipeline (`README.md`, `HANDOFF.md`, `docs/tech_design.md`, `plan-textDedup.prompt.md`).
  全面刷新架构文档以匹配当前链路（`README.md`、`HANDOFF.md`、`docs/tech_design.md`、`plan-textDedup.prompt.md`）。
- Updated `docs/cpp_interface_contract.md` to clarify that SimHash/C++ interfaces are experimental and not part of the default query flow.
  更新 `docs/cpp_interface_contract.md`，明确 SimHash/C++ 接口属于实验能力，不属于默认查询主流程。
- Updated `README.md` to reflect current two-stage architecture (`TF-IDF -> optional SBERT`) and offline SBERT usage.
  更新 `README.md`，反映当前二阶段架构（`TF-IDF -> 可选 SBERT`）与离线 SBERT 用法。
- Updated `HANDOFF.md` with implemented modules, real status, and 2026-03 retrieval tuning findings.
  更新 `HANDOFF.md`，补充已实现模块、真实状态与 2026-03 检索调优结论。
- Updated `docs/tech_design.md` with current two-stage architecture and chunking strategy conclusions.
  更新 `docs/tech_design.md`，补充当前二阶段架构与切片策略结论。
