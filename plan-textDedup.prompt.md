## Plan: TextDedup 离线检索迭代计划（更新版）

目标是持续优化当前离线检索链路：`TF-IDF -> (optional) SBERT`，并完善 `sbert-only` 模式的工程可用性与评测体系。

## 1. 当前基线

- 主链路：`two-stage`（TF-IDF + SBERT）
- 可选链路：`sbert-only`
- 已有能力：
  - 切片建库
  - 进度输出
  - two-stage 缓存
  - sbert-only 向量缓存

## 2. 近期目标（2-4 周）

1. 评测体系化
- 建立统一评测集（重复/改写/非重复）
- 输出 Recall@K、MRR、延迟、缓存命中率

2. 参数模板化
- 为不同语料沉淀默认模板（速度优先/召回优先/语义优先）
- 固化更新流程，避免拍脑袋调参

3. 批量查询化
- 增加 query 文件列表与 stdin 批处理入口
- 支持批量结果导出与复核

4. 可解释性增强
- 输出阶段性分数与命中依据
- 增加命中片段定位与高亮范围

## 3. 配置建议（vNext）

- `retrieval_mode = two-stage`
- `enable_tfidf = true`
- `enable_sbert = true`
- `tfidf_candidate_k = 800`
- `sbert_top_n = 20`

语义优先场景：

- `retrieval_mode = sbert-only`
- `reuse_sbert_cache = true`

## 4. 工程验收清单

1. `pytest -q` 全量通过
2. 默认配置可直接运行
3. 缓存命中时能够观察到明确进度日志
4. 文档一致性检查通过（README / HANDOFF / tech_design）

## 5. 非目标

- 在线服务化
- 分布式实时检索
- 数据采集流程整合入仓库
