# FlashMLA 基础 PD 修补与验收

本补丁已同步 fork #16 `5a466d655719813811ca9271e5885ffd50943c84`，
配套 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
当前为 Draft：本地 CPU 回归可验证字节范围和控制流，真实权重 P→D、RDMA、NPU kernel 和性能尚未验收。
其中，#16 的 DSpark profiling 修复和 RMSNorm 自定义算子回退均已继承。
发布机在该 #16 提交上跑通了真实权重四机 graph + DSpark 混部；GPQA 仍有客户端补题记录，
这不替代本分支的 PD 验收。本轮用户选择先交付基础 PD，PD + DSpark 随后接入。

A5 的 `enable_custom_op()` 返回关闭时，公共 `AscendRMSNorm` 必须调用原生
`torch_npu.npu_add_rms_norm` 并保留已加载的 bias，不能仍然调用不可用的 `AddRmsNormBias`。
P/D 的 target 模型也经过该公共算子，因此即使本轮关闭 DSpark，仍需要继承这项修复。
CPU 分支回归覆盖 FP32/BF16/FP16、bias 未分配/未加载/已加载、residual 有无及自定义算子开关；
这些替身测试不证明真实 NPU kernel 数值。

## 原因和改动来源

Kimi-K3 混合 MLA 与 KDA 状态。融合 MLA 的有效 payload 可能短于物理 block stride；
把整个 stride 当成传输长度，会写到 descriptor 中其他层或 padding。
本地 1 号 `de31c53dc5b94ff246b17aa198404a082162c2f9` 的单 view 排除规则提供了参考。
本补丁进一步要求多 view 只有连续、无空洞且不重叠的有效范围才可合并，保留每端独立 stride 寻址。

原有 hybrid 传输失败仍报“接收完成”，块 ID 错误接口无法区分不同 cache group。
本补丁接入配套 vLLM 已有的 `KVConnectorTransferResults.failed_recving`，失败和完成同次返回；
跨 rank 汇总等待所有 rank 完成，调度器随后重算或报错。失败也 ACK，释放 P 保留页。
这部分不能直接照搬 1 号，因为 1 号也没有接好该请求级失败接口。

新握手携带协议版本、每层类型、cache group 编号、组件 dtype、有效 shape/inner stride、块比例和 payload 长度。
两端都要使用本补丁。格式不匹配在提交 READ 前拒绝；不同物理容量/stride 本身不构成格式不匹配。
上游 #16456 `4ef088d5` 的 component-major 放行不能直接替代 token-fused FlashMLA 的握手；
上游 #14340 `51e68897` 的主线同步也不应整体覆盖 #16。

## 与本地 1 号的逐项复核

参考固定为本地 `va-k3-oldmain` 的 `de31c53`，未用远端同名方案替代。
以下是当前代码关系，不能据此宣称完成设备验收。

| 检查项 | 当前 #17 与 1 号的关系 |
| --- | --- |
| connector 注册 | `MooncakeConnectorV2` / `MooncakePullConnector` 注册入口已存在，与 1 号一致，无需重复注册 |
| 单融合 view、payload/stride | 保留 1 号单 view 不合并规则；进一步限制多 view 的有效字节并集，寻址各用 P/D 自己的 stride |
| 注册 backing、descriptor 与块比例 | 沿用配置驱动的 backing 注册；已有真实 planner/allocator 的 CPU 跨块拷贝与邻层保护回归 |
| P 末 token 截断 | 已接入；比 1 号提前到 `on_new_request`，在本地 prefix 查询之前更新 prompt，防止旧长度命中计数 |
| D 的 MLA/KDA 块映射 | 复用已有逻辑：MLA 展开 manager/kernel 块比例，KDA 取 P 的末状态页和 D 的目标状态页；首版 speculative 关闭 |
| 传输失败和 P 页释放 | 补齐 1 号未接的请求级 `failed_recving`；跨 rank 完成后重算/报错，并 ACK 释放 P 保留页 |
| cache group 编号一致性 | 本轮补齐：同名层 payload 一致但 P/D group 次序不同也必须在 READ 前拒绝 |
| DSpark、异构 TP/CP、KVPP | 不整体移植 1 号的拓扑能力；基础 PD 首版仍明确拒绝这些组合 |

本轮确认的遗漏在 `_validate_fused_mla_layout`：旧校验没有比较 `group_indices`，
而 `_build_transfer_block_buckets` 用 D 的 group 编号同时索引两端请求的 block table。
因此即使 dtype/shape/stride 全相同，P/D 分组次序不同仍可能读取另一组的块号。
修补在握手阶段逐层核对 group 编号；不扩大首版支持范围，也不改变相同分组时的传输路径。
新增 CPU 回归交换 MLA/KDA group 编号，旧代码接受，修补后在构建远端传输布局时拒绝。
1 号也采用相同编号假设，所以这项不是照搬漏掉的代码，而是对现有假设补校验。

## 首版范围

| 组合 | 状态 |
| --- | --- |
| A5 MLA_FLASH 同构，MRV2，MooncakeConnectorV2 / MooncakePullConnector | 已接代码；待真机验收 |
| 同 TP/PP，PCP=DCP=1，BF16/FP16 非量化 KV | 首版允许范围 |
| BLHNC/LBHNC/LBNHC，manager/kernel 比例 1/3/6 | CPU allocator→register→地址生成→字节保护回归 |
| MLA + KDA，失败重算/报错、跨 rank 失败、ACK | CPU 控制流回归 |
| MRV1、KVPP、自定义 connector、异构 TP/PP/CP | 拒绝 |
| PD + DSpark / 其他 speculative | 拒绝；后续独立验证 draft/hidden-state/回滚 |
| graph、EP、flashcomm1、真实权重质量和吞吐 | 本补丁未做设备验证 |

权重 W4A8 与 KV 量化不同，不因权重 W4A8 拒绝。
模型长度沿用已验收 #16 配置；本补丁没有证明新的最大长度或 128k × bs16 容量。

## 本地检查

在仓库根目录、已有 CPU torch/pytest 环境中运行：

```bash
python -m pytest --confcutdir=tests/ut/worker/v2 \
  tests/ut/worker/v2/test_flashmla_pd.py \
  tests/ut/worker/v2/test_hybrid_descriptor_layout.py \
  tests/ut/worker/v2/test_hybrid_state_page_layout.py \
  tests/ut/worker/v2/test_flashmla_phase_contract.py
python -m pytest --confcutdir=tests/ut/ops \
  tests/ut/ops/test_rmsnorm_fallback_contract.py
```

这些测试执行生产函数及固定 ced6857 的 collector/aggregator/scheduler 源码片段；
设备、部分 spec/config、网络引擎用替身，CPU ctypes 代替 RDMA。
`fixtures/ced685_pd_sources.json` 记录来源路径、文件 hash 和原文。
完整 Mooncake UT（含 msgpack 往返）须在已有 vLLM/msgspec/NPU 依赖的验证环境运行：

```bash
pytest tests/ut/kv_offload/mooncake_v2/
bash format.sh ci
```

## 设备交接顺序（本轮未执行）

1. 先记录 #16 真实权重混部验收的 VA/vLLM SHA、镜像、模型路径、配置和结果；
   若 #16 后续更新，核对 allocator/descriptor/connector 接口，重放 CPU 回归后才换底。
2. 验证负责人分配两组 A5，P/D 使用同一 PD 提交、权重、TP/PP、KV dtype、block size。
   先关闭 speculative 和 KVPP，PCP/DCP=1，使用 eager，保留原环境的网络配置。
3. 在 `/workspace` 沿用已验收 #16 的完整 `vllm serve` 命令，各端设置
   `VLLM_USE_V2_MODEL_RUNNER=1`、`VLLM_ASCEND_ENABLE_FLASH_MLA=1`，加入 `--enforce-eager`；
   P 加 `--kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":37000}'`，
   D 加 `--kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":37010}'`。
   API/side-channel 端口与实际 IP 从负责人分配取值，不复用被占用设备或端口。
4. 使用仓库 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` 对接 P/D；
   另保留同配置混部参考端。单独记录真正发生 remote READ 的日志/计数，不能用 HTTP 200 代替。
5. P 的 KDA prompt 末 token 截断及 D 首次短 prefill 继续按既有逻辑处理；核对首 token 和后续多 token，
   不把 consumer 或 query_len=1 强制判为 FlashMLA decode。

准备同 tokenizer 的 token-ID prompt cases JSON，例如 `[{"name":"boundary-128","prompt":[...]}]`。
覆盖长度 1/2/127/128/129/383/384/385/767/768/769，按实际 block size 扩展；
加无/部分/全 prefix 命中、chunked prefill、P/D 不同块号、末块、短长并发和多轮页复用。
真实 token ID 应来自模型 tokenizer，不能直接提交示例中的省略号。

```bash
python tests/e2e/manual/flashmla_pd_correctness.py \
  --baseline http://BASELINE_HOST:BASELINE_PORT \
  --pd http://PROXY_HOST:PROXY_PORT --model SERVED_MODEL_NAME \
  --cases /workspace/pd-cases.json --output /workspace/pd-results.json \
  --concurrency 1 --repeat 2 --atol AGREED_LOGPROB_TOLERANCE
```

该工具只访问已启动的 API，比较生成 token、finish reason 和逐 token logprob；不会启动服务。
数值容差先由参考端重复运行确定，再固定。报告在请求发出前列出全部 case/repeat 编号，
逐条落盘成功、失败和未完成状态；单条失败不会停止其他结果的收集，失败或缺项时退出码非零。
先核对 expected/completed/passed/failed 数量，不能把部分成功或健康检查当成全量验收。
输出一致仍须结合传输日志排除 D 全量本地重算。
补充 NPU 接收快照和邻层/邻页/padding guard；注入 READ 超时、单 rank 失败、取消和布局不匹配，
验证 D 重算/报错、不会使用部分状态，P ACK/超时回收不泄漏，也不会在 READ 未结束时提前释放。
最后开启 graph 与 eager 对照，记录完整质量评测和 TTFT/TPOT/吞吐；这些完成前不要转为已验收。
