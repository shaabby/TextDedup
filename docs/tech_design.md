# TextDedup 技术文档

## 1. 文档范围

本文仅覆盖当前 Git 仓库已交付模块：

- `src/textdedup`：文本相似度与两阶段检索
- `text-catcher`：站点递归采集

说明：`text-plagiarism-dataset` 不在当前仓库提交范围内，本文件不展开该部分设计。

## 2. TextDedup 算法原理

### 2.1 TF-IDF + 余弦相似度

- 将文本分词后映射为 TF-IDF 向量
- 使用余弦相似度衡量语义接近程度：

$$
sim(A, B) = \frac{A \cdot B}{\|A\|\|B\|}
$$

- 优点是解释性强、落地快，适合原型验证

### 2.2 SimHash

- 对 token 哈希后按词频加权叠加
- 对各维符号取值生成定长指纹（默认 64 bit）
- 用汉明距离估计相似度：

$$
sim = 1 - \frac{d_H(fp_A, fp_B)}{n}
$$

### 2.3 Two-Stage 检索

- Stage 1：用 SimHash 做候选过滤，缩小搜索空间
- Stage 2：对候选集做 TF-IDF 余弦精排
- 目标是在精度和效率之间取得平衡

## 3. text-catcher 采集设计

### 3.1 抓取流程

- 从 `targets.domains` 生成种子 URL
- 使用队列 `deque` 进行 URL 递归遍历（广度优先）
- 请求成功后提取正文文本并写入 JSONL
- 在配置条件下发现并入队下一层链接

### 3.2 约束与过滤

- 仅处理 `text/html` / `application/xhtml+xml`
- 支持 `allowed_domains` 限域
- 支持 `path_allow_patterns` 与 `path_request_only_patterns`
- 跳过常见二进制/静态资源 URL
- 通过 URL hash + 内容 hash 做增量去重

### 3.3 状态落盘

- `collect_state.json` 持久化已完成 URL hash
- 支持中断后继续采集，避免重复抓取

## 4. 依赖说明

- `numpy`：数值计算
- `scikit-learn`：TF-IDF 与余弦相似度
- `faiss-cpu`：后续向量检索扩展
- `onnxruntime`：后续模型推理扩展
- `pytest`：测试框架
- `beautifulsoup4` / `pyyaml` / `requests`：采集模块依赖

## 5. 后续迭代建议

1. 为 `text-catcher` 增加单元测试与集成测试。
2. 将采集模块对外部兄弟目录的运行时依赖收敛到仓库内。
3. 接入句向量模型并补充离线评测指标（P/R/F1、召回@K）。
