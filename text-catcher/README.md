# text-catcher

Collect-only crawler project with recursive site crawling.  
仅用于采集数据的爬虫项目，支持站内递归抓取。

It reuses scraper/storage modules from sibling project `text-plagiarism-dataset` and only writes raw documents.  
它复用了兄弟项目 `text-plagiarism-dataset` 的抓取与存储模块，并且只写入原始文档。

No generation pipeline is used (no synonym replacement, no paraphrase, no shuffle, no back-translation).  
不使用任何生成流水线（不做同义词替换、不做改写、不做打乱、不做回译）。

Only HTML page text is collected. Binary resources (images/videos/files/CSS/JS) are skipped.  
仅采集 HTML 页面文本，二进制资源（图片/视频/附件/CSS/JS）会被跳过。

## Structure

- `src/text_catcher/collector.py`: collect entry point / 采集入口
- `config.yaml`: targets/crawl/storage settings / 目标域名与抓取配置
- `scripts/collect.sh`: one-click collect script / 一键采集脚本
- `data/raw/raw_docs.jsonl`: collected documents / 采集结果
- `data/raw/failed_urls.jsonl`: failed URL log / 失败 URL 日志
- `data/raw/collect_state.json`: incremental state / 增量采集状态
- `docs/site-crawl-spec.md`: requirement spec / 需求说明

## Run

From this folder:  
在当前目录执行：

```bash
bash scripts/collect.sh
```

## Crawl scope control

Domain targets are configured in `targets.domains` only.  
目标站点仅通过 `targets.domains` 配置。

Configure in `config.yaml` under `crawl`:  
在 `config.yaml` 的 `crawl` 中配置：

- `connect_timeout_seconds`: max wait for connection establishment / 连接建立最大等待秒数
- `checkpoint_every_n_urls`: state checkpoint batch size / 状态批量落盘间隔（按 URL 数）
- `recursive_enabled`: enable/disable recursive crawling / 是否开启递归
- `allowed_domains`: allowed domains (empty means infer from seed URLs) / 允许抓取域名
- `max_depth`: max recursion depth / 最大递归深度
- `max_pages`: max HTML pages to store per run / 单次运行最多落盘页面数
- `follow_query_strings`: whether query string is part of URL dedup / 去重是否包含 query
- `only_html`: keep true to process HTML only / 仅处理 HTML
- `skip_binary_resources`: keep true to skip image/file/static URLs / 跳过图片与静态资源
- `path_allow_patterns`: request + store patterns / 发请求且入库的路径规则
- `path_request_only_patterns`: request-only patterns / 只发请求不入库的路径规则
- `seed_always_request`: always request depth-0 seed page / 首页种子页始终可请求

Resume behavior: crawler state now persists both completed URL hashes and pending queue.  
断点续跑行为：状态文件会同时保存“已完成 URL 哈希”和“待处理队列（pending queue）”。

Example (`x` is any segment):  
示例（`x` 为任意片段）：

```yaml
crawl:
	path_allow_patterns:
		- /show/*/*.html
```

Request policy:  
请求策略：

- Matches `path_allow_patterns`: request + store.
- Matches `path_request_only_patterns`: request only, no store.
- Matches neither: do not request.

Priority rule: when a path matches both groups, `path_allow_patterns` wins (the page is stored).  
优先级规则：当路径同时匹配两组时，`path_allow_patterns` 优先（页面会入库）。

- 匹配 `path_allow_patterns`：发请求并入库。
- 匹配 `path_request_only_patterns`：只发请求不入库。
- 两者都不匹配：不发请求。

## How reuse works

`collector.py` appends this sibling path to `sys.path`:  
`collector.py` 会把以下兄弟目录加入 `sys.path`：

`../text-plagiarism-dataset/src`

Then imports and uses:  
然后导入并复用：

- `scrapers.base.BaseScraper`
- `scrapers.base.Document`
- `storage.dataset.JsonlStorage`

So `text-catcher` remains collect-only, but shares the hardened crawler foundation.  
因此 `text-catcher` 依然是纯采集项目，同时共享了已加固的爬虫基础能力。

## Recent Changes Summary

- Unified recursive crawler architecture: removed source-specific scraper branching and switched to domain-driven crawling via `targets.domains`.
- HTML-only policy: only `text/html`/`application/xhtml+xml` responses are processed; binary/static resources are skipped.
- Path policy split:
	- `path_allow_patterns`: request + store
	- `path_request_only_patterns`: request only (no store)
	- unmatched paths: do not request
- Path priority fix: when a URL matches both groups, `path_allow_patterns` wins.
- Incremental dedup:
	- skip already completed URLs through `completed_url_hashes`
	- avoid duplicate enqueue using in-memory hash sets
- Batched checkpoints: added `checkpoint_every_n_urls` to reduce state write overhead.
- Resume reliability improvement: `collect_state.json` now persists both:
	- `completed_url_hashes`
	- `pending_queue`
	so interrupted runs can resume from frontier queue instead of restarting from seeds.
- Logging improvements:
	- `[connected]` on successful HTTP response
	- `[collect]` on successful storage
	- `[checkpoint]` on batched state save
	- `[done]` on finished or interrupted-with-save exit

## 最近改动总结

- 统一递归爬虫架构：移除按来源分支的特殊 scraper，改为 `targets.domains` 域名驱动。
- 仅抓 HTML：仅处理 `text/html`/`application/xhtml+xml`，图片与静态资源跳过。
- 路径策略拆分：
	- `path_allow_patterns`：发请求并入库
	- `path_request_only_patterns`：只请求不入库
	- 两者都不匹配：不发请求
- 优先级修复：同时命中两类规则时，以 `path_allow_patterns` 为准（允许入库）。
- 增量去重：
	- 使用 `completed_url_hashes` 跳过已完成 URL
	- 使用内存哈希集合避免重复入队
- 批量 checkpoint：新增 `checkpoint_every_n_urls`，减少频繁全量写状态的开销。
- 断点续跑增强：`collect_state.json` 同时保存：
	- `completed_url_hashes`
	- `pending_queue`
	中断后可从队列续跑，而不是只从种子页重启。
- 日志增强：
	- `[connected]` 请求成功
	- `[collect]` 成功入库
	- `[checkpoint]` 批量状态落盘
	- `[done]` 正常结束或中断保存后结束
