# TextDedup Python/C++ Interface Contract (Detailed)

> 状态说明（2026-03）：
> 当前主查询链路为 `TF-IDF -> (optional) SBERT`，不依赖 SimHash。
> 本文档中的 SimHash/C++ 接口定义仅用于后续性能实验与兼容模块，不是当前主流程契约。

## 1. 目标与范围

本文定义 TextDedup 在后期进行 C++ 性能增强时的接口契约，目标是避免“边做边猜”导致的接入失败。

适用范围：

- Python 编排层与 C++ 计算层之间的数据交换
- C++ 模块导出函数签名
- 错误模型、版本策略与回退机制
- 一致性测试与验收标准

当前不适用范围（主流程）：

- `scripts/query_similar_segments.py` 的默认 `two-stage` 查询路径
- 生产级/默认参数配置文件中的必选能力

不在范围内：

- 在线服务协议
- 前端/UI 协议
- 模型推理框架内部实现细节（如 SBERT 推理引擎）

## 2. 架构边界

职责分层（实验路线）：

- Python 层：配置、数据加载、流程编排、日志、报告输出
- C++ 层：纯计算热点实验（例如 SimHash 批量计算、汉明距离批量 TopK、可选片段对齐）

硬约束：

- C++ 不直接读配置文件
- C++ 不做业务规则判定
- C++ 不触碰磁盘 I/O（除非未来单独定义缓存接口）

## 3. 数据契约（Schema）

### 3.1 通用类型

- `DocId`: `str`（UTF-8）
- `Token`: `str`（UTF-8）
- `Weight`: `float32`
- `Fingerprint`: `uint64`
- `Score`: `float32`

### 3.2 Python 侧标准数据模型（建议）

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class WeightedDocument:
    doc_id: str
    tokens: list[str]
    weights: list[float]  # must align with tokens by index

@dataclass(frozen=True)
class SimHashFingerprint:
    doc_id: str
    hash_bits: int        # currently 64 only
    fingerprint: int      # uint64 in Python int wrapper

@dataclass(frozen=True)
class HammingCandidate:
    doc_id: str
    distance: int
    score: float          # normalized score, e.g. 1 - distance/hash_bits

@dataclass(frozen=True)
class EngineError:
    error_code: str
    error_message: str
    stage: str

@dataclass(frozen=True)
class EngineMeta:
    engine: str           # "python" | "cpp"
    version: str          # e.g. "simhash_v1"
    latency_ms: float
    fallback_used: bool

@dataclass(frozen=True)
class TopKResult:
    query_doc_id: str
    candidates: list[HammingCandidate]
    meta: EngineMeta
    error: Optional[EngineError] = None
```

### 3.3 输入输出约束

输入约束：

1. `tokens` 与 `weights` 等长且大于 0
2. `doc_id` 非空且在批次内唯一
3. `hash_bits` 当前仅允许 64
4. `top_k >= 1`

输出约束：

1. `candidates` 按 `distance` 升序，`distance` 相同按 `doc_id` 升序（保证可复现）
2. `score` 为确定性函数，不依赖运行时随机状态
3. 任意失败场景必须返回结构化错误，不允许裸异常直接泄漏到业务层

## 4. C++ 导出接口（建议签名）

建议使用 `pybind11` 封装以下函数，函数名带版本后缀。

```cpp
// v1: weighted simhash for a batch of docs
std::vector<uint64_t> compute_simhash_v1(
    const std::vector<std::vector<std::string>>& token_batches,
    const std::vector<std::vector<float>>& weight_batches,
    int hash_bits // must be 64 in v1
);

// v1: top-k nearest by hamming distance
std::vector<std::vector<std::pair<int, int>>> hamming_topk_v1(
    const std::vector<uint64_t>& query_fps,
    const std::vector<uint64_t>& corpus_fps,
    int top_k,
    int max_distance // inclusive threshold
);

// Optional in later phase: alignment explain kernel
std::vector<std::vector<std::pair<int, int>>> align_spans_v1(
    const std::vector<std::string>& query_tokens,
    const std::vector<std::string>& cand_tokens,
    float min_match_score
);
```

说明：

- `hamming_topk_v1` 返回 `(corpus_index, distance)`，Python 侧映射 `doc_id`
- 若未来支持 128-bit，新增 `*_v2`，不修改 `v1` 语义

## 5. Python Bridge 约定

建议新增 `src/textdedup/cpp_bridge.py`，对外仅暴露稳定接口。

```python
class CppBridge:
    def is_available(self) -> bool: ...

    def compute_simhash(
        self,
        docs: list[WeightedDocument],
        hash_bits: int = 64,
    ) -> list[SimHashFingerprint]: ...

    def hamming_topk(
        self,
        query_fps: list[SimHashFingerprint],
        corpus_fps: list[SimHashFingerprint],
        top_k: int,
        max_distance: int,
    ) -> list[TopKResult]: ...
```

bridge 行为约束（实验模块）：

1. 首选 C++，失败自动 fallback Python
2. fallback 必须记录 `meta.fallback_used = true`
3. fallback 后结果结构与排序必须保持一致

## 6. 错误码规范

错误码命名：`TD_<STAGE>_<TYPE>`。

建议首批错误码：

- `TD_IO_INVALID_INPUT`: 输入为空、长度不一致、参数越界
- `TD_SIMHASH_UNSUPPORTED_BITS`: hash_bits 非 v1 支持范围
- `TD_CPP_NOT_AVAILABLE`: 扩展未安装或加载失败
- `TD_CPP_RUNTIME_ERROR`: C++ 执行异常
- `TD_CPP_TIMEOUT`: C++ 任务超时（若后续引入）
- `TD_FALLBACK_FAILED`: C++ 失败且 Python fallback 也失败

错误处理规则：

1. C++ 内部异常统一映射为 `TD_CPP_RUNTIME_ERROR`
2. Python 业务层不得依赖异常文本做逻辑判断，只可依赖 `error_code`
3. `error_message` 面向调试，不作为契约字段做强匹配

## 7. 版本与兼容策略

版本规则：

1. 接口与语义变更使用函数后缀版本（`*_v1`, `*_v2`）
2. 新增字段采用“只增不改”，旧字段不可重命名
3. 删除字段必须先经历一个完整 deprecation 周期

兼容矩阵建议：

- Python bridge 声明支持的最小 C++ 扩展版本
- 启动时做版本探测，不兼容则自动切 Python

## 8. 性能与一致性验收

### 8.1 一致性（先决）

必须先通过一致性，再谈性能。

一致性指标：

1. 指纹一致率：`compute_simhash_v1` 与 Python 实现逐条相等
2. TopK 一致率：候选集合一致，排序规则一致
3. 分数容差：浮点分数误差 `<= 1e-6`

### 8.2 性能（后置）

离线批处理场景建议验收门槛：

1. SimHash 批量计算阶段耗时下降 `>= 30%`
2. 汉明 TopK 阶段耗时下降 `>= 40%`
3. 端到端总耗时下降 `>= 20%`

## 9. 测试计划（必须项）

建议新增 `tests/test_cpp_parity.py`，覆盖：

1. 正常路径：小样本与中样本 parity
2. 边界路径：空 token、超长 token、重复 doc_id
3. 错误路径：模拟 C++ 加载失败，验证自动 fallback
4. 稳定性：同输入多次运行结果完全一致

建议新增性能基准脚本 `tests/bench_cpp.py`：

1. 固定随机种子
2. 固定语料规模（例如 1k/10k/50k）
3. 输出 Python/C++ 分阶段耗时对比

## 10. 实施里程碑（2 周 PoC）

Week 1（实验阶段）:

1. 冻结契约：schema、函数签名、错误码
2. 完成 `cpp_bridge.py` 空实现 + Python fallback
3. 完成 C++ `compute_simhash_v1` PoC

Week 2（实验阶段）:

1. 完成 C++ `hamming_topk_v1` PoC
2. 补齐 `test_cpp_parity.py` 与 `bench_cpp.py`
3. 形成 PoC 结论：是否进入正式 C++ 化

## 11. 非目标（当前阶段）

- 不进行全仓库 Python -> C++ 重写
- 不改造 SBERT 模型推理内核
- 不引入在线服务协议或 RPC
- 不将 SimHash/C++ 实验接口回灌到当前默认检索主链路
