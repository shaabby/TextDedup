# TextDedup Handoff

## 1. Repository Scope

Workspace root:
- `/home/user/code/TextDedup`

当前 Git 仓库交付范围包含：
- `TextDedup` 主模块（相似度与两阶段检索）
- `text-catcher` 子模块（仅采集的递归抓取工具）

不包含：
- `text-plagiarism-dataset`（如本地存在，仅作为外部兄弟目录依赖，不随本仓库提交）

## 2. TextDedup 主模块

### 2.1 Implemented Modules

- `src/textdedup/similarity.py`
  - `SimilarityEngine` (TF-IDF + cosine similarity)
  - `pairwise_similarity`
- `src/textdedup/simhash.py`
  - `SimHash` fingerprinting and Hamming-distance similarity
- `src/textdedup/two_stage.py`
  - `TwoStageSearchEngine`
  - Stage 1: SimHash candidate filtering
  - Stage 2: TF-IDF cosine reranking
- `src/textdedup/__init__.py`
  - Exports: `SimilarityEngine`, `pairwise_similarity`, `SimHash`, `TwoStageResult`, `TwoStageSearchEngine`

### 2.2 Tests

- `tests/test_similarity.py`
- `tests/test_simhash.py`
- `tests/test_two_stage.py`

运行：

```bash
/home/user/code/TextDedup/.venv/bin/python -m pytest -q
```

## 3. text-catcher 采集模块

路径：
- `text-catcher/`

关键文件：
- `text-catcher/src/text_catcher/collector.py`：递归抓取入口
- `text-catcher/config.yaml`：站点与抓取策略配置
- `text-catcher/scripts/collect.sh`：一键采集脚本
- `text-catcher/data/raw/raw_docs.jsonl`：采集结果

当前行为：
- 仅抓取 HTML 页面文本
- 支持基于 `deque` 的广度优先递归抓取
- 支持域名白名单、路径规则、URL 去重与失败落盘

运行：

```bash
cd /home/user/code/TextDedup/text-catcher
bash scripts/collect.sh
```

## 4. Integration Note

`text-catcher` 当前通过 `sys.path` 动态引入兄弟目录中的模块（如 `storage.dataset`、`scrapers.base`）。如果目标环境不存在 `../text-plagiarism-dataset/src`，采集程序会在启动时抛出依赖路径缺失错误。

## 5. Recommended Next Steps

1. 为 `text-catcher` 增加独立测试（URL 规范化、路径策略、递归队列行为）。
2. 将 `text-catcher` 中对外部兄弟目录的依赖内聚到本仓库，减少部署耦合。
3. 在根目录补充统一 `Makefile` 或任务脚本，简化测试与采集入口。
