# TextDedup

TextDedup 是一个面向文本去重与相似检索的工程化原型仓库。

当前仓库主要包含两个可交付模块：

- `src/textdedup`：文本相似度与两阶段检索算法原型
- `text-catcher`：仅采集（collect-only）的站点递归抓取工具

说明：`text-plagiarism-dataset` 在本地开发时可能作为兄弟目录存在，但不属于当前 Git 仓库交付范围，上传时不包含该目录。

## 快速开始

### 1. TextDedup 算法模块

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

### 2. text-catcher 采集模块

在仓库根目录执行：

```bash
cd text-catcher
pip install -r requirements.txt
bash scripts/collect.sh
```

## 目录说明

- `src/textdedup`：`SimilarityEngine`、`SimHash`、`TwoStageSearchEngine`
- `tests`：相似度与两阶段检索单元测试
- `docs`：根项目技术设计与交接文档
- `text-catcher`：配置驱动的递归 HTML 采集工具
- `data`：根项目预留数据目录
