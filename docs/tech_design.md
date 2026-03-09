# TextDedup 技术文档（本周）

## 1. 算法原理

### 1.1 TF-IDF + 余弦相似度

- 将文本分词后映射为 TF-IDF 向量
- 使用余弦相似度衡量语义接近程度：

$$
\text{sim}(A, B) = \frac{A \cdot B}{\|A\|\|B\|}
$$

- 适合原型阶段快速验证，解释性强

### 1.2 SimHash

- 对 token 计算哈希并按词频加权累加
- 最终根据向量符号位生成定长指纹
- 两文本相似度可由汉明距离估计：

$$
\text{sim} = 1 - \frac{d_H(fp_A, fp_B)}{n}
$$

其中 $d_H$ 为汉明距离，$n$ 为指纹位数（默认 64）。

## 2. 依赖说明

- numpy：数值计算
- scikit-learn：TF-IDF 与余弦相似度
- faiss-cpu：后续向量检索索引
- onnxruntime：后续模型推理加速
- pytest：单元测试

## 3. 已完成能力

- 两两文本相似度计算
- 基于阈值的批量去重对检出
- 文本查询 Top-K 相似检索

## 4. 后续迭代建议

- 接入句向量模型（如 BGE/E5）并导出 ONNX
- 用 Faiss 替代 sklearn 检索做大规模加速
- 增加评测集与精确率/召回率指标
- 提供 Python SDK 与 C++ 动态库统一接口
