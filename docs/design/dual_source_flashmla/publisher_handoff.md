# PR 15 发布机验证交接

本轮优先验收 #15；#13 `93dbf8f91c0ca3d94e5617b0a1903b0456edc40a`
保留为仍有异常的对照。VA 使用本次交付回报中的完整 SHA，配套 vLLM 固定为
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。不重合移动的上游 head，
不只记录分支名。本文是待执行步骤；本地未运行 NPU、构建或服务。

本次补齐混合历史 FIA、KDA 脏页复用测试与延迟取证，按
[逐步验证清单](followup_validation.md)执行。此前短窗口说明仍适用于不设置
`arm_file` 的运行；第七轮定位必须保留同一服务的前六轮状态。

## 1. 锁定实际运行来源

在发布机原有 checkout 正常拉取 `codex/dual-source-flashmla`，核对交付 SHA、
vLLM SHA、工作区改动与实际 Python import 路径。记录 torch/torch_npu、CANN、
FLA wheel/native OPP、外部 FlashMLA wheel/OPP 的版本与文件 hash。
保留原机器启动命令、模型配置及硬件拓扑；不要同时更换依赖以免失去对照。

## 2. 先复现同一启动条件

沿用最新 #13 失败回执的 dummy 5 层、eager、TP8/DP1、无 speculative 完整命令；
MRV2 与 FlashMLA 开关开启。本文不补造未知模型路径或启动参数。
在初始化日志/调试记录中保存真实 spec、planner descriptor 和最终 tensor：

- 本卡 `num_query_heads=12`，`spec.num_heads=num_kv_heads=1`。
- 同原事故 LBNHC、53733 blocks、768 manager tokens、BF16 latent512+rope64
  时，page/block pitch 应为 `884736`，layer stride 为 `47539519488`；
  不应再出现 `10616832` 和 `570474233856`。
- 可用内存变化会改变 num_blocks/layer stride；此时按实际 block 数推导，
  不硬套原 layer stride。记录 offset、storage_offset、shape/stride、data_ptr、
  页内 payload 边界及 manager/kernel 比例。
- 初始化、健康检查和请求 smoke 分别记录；dummy 输出不用于语义正确性验收。

## 3. NPU 地址与算子接续

先执行已有设备测试：

```bash
python -m pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_hybrid_page_state_npu.py
```

测试包括实际 conv/chunk/recurrent 对 dense state 对照、selected state IO、
COW/zero，以及 writer → FIA → KDA 更新 → external FlashMLA 读取。
记录每个 case、算子路由、容差及失败原始数据。该设备 fixture 使用 dense
per-layer descriptor，不能冒充三种布局的设备全覆盖。

进一步使用运行时真实 planner 生成的 BLHNC/LBHNC/LBNHC 和 ratio1/3/6，
检查不同合法 manager ID 的状态隔离：MLA 持有块1、KDA 更新块2时，
MLA 原始 payload 必须不变。相同 manager ID 跨组别名是设计行为，不能当成
同时持有的两个独立页。多层、非零 layer/backing offset 也须记录。
复查旧 slot3 / MLA page174 事故：旧 state-major 寻址会碰到该页，修复后
真实 conv/recurrent/prefill 对 slot3 的更新不得改写不属于它的活跃 MLA 页。

BLHNC 的 MLA 子页为 `[manager, subpage, layer, payload]`，KDA 保持原
manager pitch；writer、FIA、external reader 必须使用同一个最终存储视图。
COW 先保存所有 source payload；zero 只清目标 payload，保护其他层、间隙、
其他 manager 和源页。检查实际写入字节与读回数值，不能只看 shape。

## 4. 真实权重、并发与数值验收

启动通过后再加载真实权重，先低并发固定 seed/固定输入对照；检查 token、
logits/attention 中间结果的有限性和既定数值容差，定位首个异常 stage/rank/step。
Prefill 必须走 FIA，decode 才走外部 FlashMLA；覆盖 cold/history-backed
1/2 token、chunked/长 prefill、纯 decode、混合 batch。

低并发通过后，按既有拓扑执行 C60、多轮长短混合、块回收与 prefix/COW/zero；
至少 10 轮，保留完整请求、输出、缓存证据、延迟/吞吐与峰值内存。eager 通过后
再测图模式/padding/空行/metadata 复用和事件同步。
HTTP200、无感叹号、无 NaN 单项都不能证明正确；dummy 也不能替代真实权重。

需要隔离时做固定版本 A/B/A：A=#13 93db，B=#15 新 SHA，其余软件与语料相同；
这比较的是两套上游背景，不是只比较 descriptor 修复。若同一 #15 开关
FlashMLA，则记录实际路由变化，用于隔离外部 decode。不要把带已知启动错误的
旧 b4af 当作成功参考。失败时先保留日志和首个地址/数值偏离，不叠加新依赖。
