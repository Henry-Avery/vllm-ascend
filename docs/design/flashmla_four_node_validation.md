# 旧 B 线 PR #14：四机取证与短 profiler

本页只补充将来 NPU 发布机验证的操作说明；当前不启动、不发负载。具体部署使用单独的固定 SHA
执行单，由现场唯一 owner 接单。仓库文档的发布不代表已接单、启动或通过。
旧的本地未发送草稿和 d7 部署说明不作为本轮执行单，不恢复旧自动化或新主线任务。

## 固定版本与现场

- 仓库 Henry-Avery/vllm-ascend，分支 `codex/flashmla-chunk-diagnostics`。
- 候选完整 SHA 从 PR #14 当前固定候选栏复制到本轮回执；禁止只记录移动分支名。
- 本次追加提交的父版本是 `26f2e859cf34f14ad6f5addfedc17156b1c822a2`；旧基点 d7。
- 配套 vLLM：`84030bbe3d74d99bad477a3d2e37a973ccd8865c`。
- 历史四机 `141.61.33.{12,21,22,23}`，容器 `yyt927oldoldmain`；这些不是当前空闲证明。
- 历史复现参数：eager、TP8/DP4、chunked prefill 开、max_num_batched_tokens=1536、
  prefix caching 关、无 speculative/DSpark。保留原模型、dtype、包和拓扑。

现场先核对资源归属、当前请求和服务；保留工作区改动，逐节点记录双仓 SHA、dirty patch、
导入路径、算子包身份与实际启动命令。不在运行进程下覆盖源码，不重建容器或升级依赖。
由同一 owner 切换本任务服务，不能影响其他任务。按照[采集手册](flashmla_chunk_diagnostics.md)
传递环境变量，开启 `scan_all_mla_layers=true`，每轮使用全新目录。
默认只观察各 DP 的 TP0；其他 TP 未覆盖。

## 一轮有界矩阵

| 顺序 | 输入 | 需要的证据 |
| --- | --- | --- |
| P0 | 版本与路由 | 确认 Prefill FIA、Decode external；armed 后有 batch/end，不把启动当数值通过 |
| P1 | 原 2143-token，再按需原 2150-token payload，DP0、C1、temperature=0、seed=4242、max_tokens=256 | 首 token 附近的 forward/request/phase、首个异常张量；先捕获原失败，不先跑长 smoke |
| P2 | 2150 原件依次到 DP1/2/3，各一次，预算足够时 | 实际路由、DP counts 与副本差异；发现首错即停止扩大矩阵 |
| P3 | 同一 DP 最多 C2，长请求后续 prefill 与新短请求交错 | 必须在真实 batch 看到正历史与零历史；客户端并发不等于命中 |
| P4 | 少量页尾/短尾输入 | 以内部长度证明跨 128 页及 1-token 尾段；不按字符数推测 |

历史原件在发布控制机 `work/flashmla-new-fleet/probe/` 下的
`chunked-prefill-c1-20260928/`、`chunked-prefill-matrix-c1-20260928/`。
规范化 payload SHA256 前缀分别 `52d9a0374d92310d`、`4955f0aaa142f96d`；
读取原 JSON 计算完整哈希。历史回执见
[d7 C1 条件矩阵](https://github.com/Henry-Avery/vllm-ascend/pull/10#issuecomment-5870002634)。
同一 payload 曾通过也曾失败；当时诊断预算已耗尽，不能把更早 C60 raw NaN 当作此处已证根因。

采集最多 128 个匹配真实 forward，非 HTTP 请求数。已知内部精确 ID 才用 request_ids；
否则新进程先跑目标。预算耗尽的输出未观察；必要时最多一次明确登记的窗口重置。
拿到首错就保存，不为凑齐矩阵继续发请求。不混跑 C60/GPQA、图/spec、关 chunk 或性能实验。

## 短 profiler：目标请求前开启，结束必须停止

已按本候选 `profiler/torch_npu_profiler.py`、`worker/worker.py` 及 pinned vLLM
`profiler/wrapper.py`、`entrypoints/serve/profile/api_router.py` 核对接口。
在原服务命令增加以下参数，目录应是各节点可写的新绝对路径：

```text
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/absolute/path/to/run/profile","ignore_frontend":true,"delay_iterations":0,"max_iterations":16,"torch_profiler_with_stack":false,"torch_profiler_with_memory":false}'
```

此 Ascend wrapper 采 CPU+NPU，Level1 文本导出；不承诺 record_shapes 已接通。
`max_iterations=16` 是 worker step 上限机制，当前实现按 `> max` 停止，可能包含第 17 步，
也可能计入 dummy，不能等同 16 个输出 token。没有 step 时不能依赖自动停止。

1. 服务就绪后，目标请求前对实际服务地址 POST `/start_profile`。
2. 立即发送原失败请求，采到两段 Prefill 和开头 Decode；约 20 秒的短窗口优先。
3. 无论请求成功、失败、超时，都在客户端 finally 中 POST `/stop_profile`。
   请求仍在运行时也可停止采集；不要为了等待完整 256 tokens 长时间开 profiler。
4. 检查每个预期 worker/rank 的实际 trace 文件和服务 profiler 警告。
   HTTP 200 不证明底层采集成功，也不证明调用广播到了所有 DP。

示意接口（使用现场实际 URL/认证；不额外发模型请求）：

```bash
curl --fail --max-time 30 -X POST "$FLASHMLA_SERVICE_URL/start_profile"
# 运行原目标请求；客户端用 finally 或等价清理保障下一条 stop。
curl --fail --max-time 30 -X POST "$FLASHMLA_SERVICE_URL/stop_profile"
```

诊断开启时新 CPU 范围：`flashmla.execute/<execute_id>`、
`flashmla.prepare_attn/<forward_id>`、`flashmla.writer/<forward_id>/<layer>/<phase>`、
`flashmla.decode/<forward_id>/<layer>`。用 JSONL 的 execute_id、forward_id、rank、request、UTC
关联 trace；保留原始厂商 profiler 目录，而不只截图。
这些范围表示 CPU 调用/入队，不表示设备完成。需要从实际 kernel、stream、event 依赖
核查 metadata 生产、等待、writer、Flash 消费、release/reuse；源码 marker 本身不能证明无竞态。

同步 CPU 诊断会扰动时序，trace 不用于吞吐结论。若只在关闭诊断时复现，下一轮可固定相同输入
做诊断关闭的短 profiler 对照；不自动扩展本轮矩阵。没有设备流/等待证据的部分如实记 unknown。

## 回执与停止

按[补充复核表](flashmla_oldline_review_checklist.md)逐项解释证据与未覆盖项。
保留完整服务日志、SSE、payload 哈希、内部 request ID、诊断目录、profiler 目录及如下验收输出：

```bash
python tools/flashmla_check_diagnostics.py /absolute/path/to/collected-run \
  --expected-dp-ranks 0,1,2,3 --tp-rank 0 --require-history
```

只测 DP0 就声明其他副本未测，不能伪造四机通过。退出 0 仅表示采样证据完整且比较通过；
1 为偏差，2 为不完整。首坏记录不是根因；未命中混合空行不能排除零历史问题。
首次启动故障保存日志并停止，不反复重启/改环境。首错捕获后停止新增请求、drain 并保存证据；
服务由现场 owner 按本轮约定释放或明确保留期限。没有回执不声称已部署或 NPU 通过。
