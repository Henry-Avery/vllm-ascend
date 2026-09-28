# FlashMLA 双机与主力机 agent team 分工

## 协作规则与当前状态

本文件的技术检查表供实验设计选用，不构成运行授权。主力机/发布机的暂停、人工接管与恢复统一采用[协作协议 v1.3](../collaboration/README.md)；状态见 [PR #10](https://github.com/Henry-Avery/vllm-ascend/pull/10) 最新用户授权和回执。文档更新、提交或新 PR 均不解除暂停。

历史起点为 `698d00e7c86eb8a9175c92d384497b28f5b798eb`；`bc83ce0ed` 仅加说明；`f78fe02e8` 为 TRACE 诊断提交。这些不是当前待部署候选，不能自动重跑。

历史修复 `ae92cb43e971f9e7586855b77658101f1f248acd` 避免普通 eager 历史形状永久占用 Q/metadata 缓冲。其结果按原 SHA 保留；后续候选只由新实验单指定，不因阅读历史文档而切换。

## 责任与交付

| 角色 | 负责 | 交付与边界 |
| --- | --- | --- |
| 主力机主 agent | 拆任务、审查子代理结果、复现代码缺陷、最小修复、维护提交和 PR | 共享开发分支唯一整合入口；标清新 SHA、修改原因、受影响用例。不得把 CPU/mocked 通过写成 NPU 通过 |
| 主力机审查子代理 | 检查高并发/异步/图 buffer 生命周期、padding、target/draft 交替 | 可定位问题、代码位置、触发条件、最小复现和修复建议；只读审查，不推共享分支 |
| 主力机检查子代理 | 对真实 metadata builder 增加少量 CPU 生命周期检查 | 同容量变请求的地址稳定/内容刷新、defer、padding、mixed 隔离；不模拟“已经验证 NPU 事件” |
| 主力机工具子代理 | 准备已有服务可使用的并发探测脚本与报告格式 | 请求级时间线、错误、延迟与统计；不接管发布机启动流程，不宣称仅靠 HTTP 成功证明 Flash/tiling 命中 |
| 发布机 | 按本机已有 skill 拉起服务、核实镜像/包、真实 NPU 验证、现场诊断 | 实际命令、SHA、包/导入路径、输入和期望、首次错误/堆栈/profiler、真实命中与数值。可就地修环境；代码修复用独立分支回传主力整合 |
| 用户 | 在主力机查看对比、诊断方案、风险和阶段结果，随时调整优先级 | 常规检查和已授权诊断继续推进，不把每次操作都变成用户审批 |

上表子代理角色来自历史已授权任务，不授权新会话自行创建子代理。PR 是两机共享记录，不是远程执行通道；发布机须在获得恢复授权后由唯一执行会话操作并回报。后台监听是否恢复单独确认，不重复创建调度，也不把“写了建议”当成“发布机已执行”。

## 高并发验证的推进顺序

1. **包/算子前置检查**：实际 schema/Meta、BBND 两轴 stride/offset、mask、输出/LSE；使用已有 `tools/flashmla_probe.py`。失败先停在该层修复。
2. **单请求基础**：原基线与候选使用相同环境/输入，确认 Prefill FIA → cache → Flash Decode。覆盖短 prompt 尾段、prefix/chunked 的真实场景。
3. **并发逐档**：从 1、4、16、32 开始，最高档按发布机现有 `max_num_seqs`、显存与启动配置确定。各档请求数应超过并发数，以观察请求完成和补入，不只观察第一次 batch。
4. **请求变化**：短/长 prompt 混合、不同输出长度、请求结束后补入、prefix 命中与不命中、chunked Prefill 与 Decode 共存。使用实际输入集；不能仅以客户端并发数推断 engine batch、prefix 命中或 chunking。
5. **图变量变化**：同档位变真实请求数、长度、block table、请求顺序，再跨图档位；与 eager 对照。高并发下用 TRACE 和 profiler 检查 buffer 地址和每轮 metadata 更新。
6. **DSpark**：基础与图通过后，再加 acceptance/rejection、draft padding、target/draft 交替。DCP/PCP 扩展、PD 分离仍不混入本轮。

客户端探测只建立请求级时间线；数值验证阈值和实际 graph/spec 配置由发布机现有 skill 负责。TRACE/profiler/断点用于诊断，关闭这些干预后才做性能对比。不要在仍有首次功能错误时继续扩大压力掩盖现场。

## 发生问题时在哪里打桩

| 现象 | 主力机检查/打桩点 | 发布机必须拿到的现场 | 后续修复原则 |
| --- | --- | --- | --- |
| 并发升高后卡住 | MRv1 executor submit/wait/release；MRv2 metadata scope；长度同步处 | 各 worker 全线程 Python 堆栈、最后 metadata/replay 标记、设备/通信时间线 | 区分 host 等待、事件依赖、算子超时与通信；基于证据修生命周期，不能加全局同步把问题藏起来 |
| 地址相同但结果像旧请求 | `FlashMLAMetadataBuilder.build.refresh`、source lengths/cu、schedule copy | 同 builder/档位的 step、源和目标指针；最小设备长度/table/schedule 摘要；请求重排证据 | 检查本轮刷新、输入 ready 和上一轮 reuse fence；不要靠每轮重新分配绕过失效复用 |
| 长短混合后结果错 | 阶段 splitter、排序、Prefill metadata/context、decode token slice | CPU phase/query boundaries 与设备 lengths、各阶段真实算子事件 | 保持短 Prefill=FIA；长度修复与排序修复分别提交 |
| KV 污染或越界 | 原 Prefill/Decode writer、slots、fused cache view | backing/stride/offset、slot=-1、cache guard、首次失败算子 | 保留 2 号非连续协议；不静默 contiguous/repack 或切回 FIA |
| eager 正常，graph 错 | metadata 图外更新、图 descriptor、FIA updater 过滤 | 同图变输入的 eager/graph 对照、地址和内容变化、真实 replay 事件 | 修正具体 capture/replay 差异；不以 Python 首次日志当作设备已执行 |
| DSpark 才错 | draft context writer、draft lengths/causal、独立 executor | target/draft 对应轮次、rejection、下一轮长度与 padding | 先隔离 draft/target，最小用例复现；不无证据地全局关闭 CPU mirror |
| 只有延迟/吞吐退化 | metadata kernel、attention kernel、CPU length D2H/sync | 关闭 TRACE 的基线/候选同输入 profile，明确 host/device 时间 | 当前仍保留 CPU mirror/sync；单独提出消除同步优化，不能直接等同于 tiling 下沉无效 |

在用户当前有效授权范围内，按需使用断点、临时打桩、profile 和 Python 全线程堆栈；历史诊断授权不能覆盖后来的暂停。优先使用当地已安装工具/skill；保存现场后缩小复现。临时同步、断点暂停和诊断日志的干预必须记录。

## 两机迭代约定

发布机回报在 PR #10，目标 PR #6 同步重要结论；一次回报至少包含：

```text
实际 checkout SHA / TRACE 是否开启 / runner、graph、spec、TP 配置
完整镜像 ID / CANN、torch_npu、算子包版本和实际 import 路径
用例、输入规模、客户端并发、服务端 batch/资源情况与预期
命令 / 第一次失败 / 完整 traceback / 最后成功标记
请求时间线、各 worker Python 堆栈和 profiler 日志路径
临时修改或独立修复分支及完整 SHA / 复测结果
需要主力机处理的明确问题
```

主力机收到证据后：审查代理定位 → 主 agent 选最小修复 → 检查代理补回归（能在 CPU 证明的部分）→ 独立签名提交 → PR 指定新的运行 SHA 和重测用例。功能修复、诊断桩、压测工具/测试分别提交；主力机整合发布机独立修复前先 fetch 和核对差异，不 force push。

发布机只对明确指定的新 SHA 做受影响用例复测，保留此前版本结果，不把移动的 branch head 作为测试版本。没有错误也要交付算子命中和数值证据；HTTP 成功、mock 测试、日志、NPU kernel、图、完整服务和性能结果分层报告。

参考：[诊断手册](flashmla_diagnostics.md)、[CPU length/图更新对比](flashmla_plan1_comparison.md)、[开发记录](flashmla_development_status.md)。
