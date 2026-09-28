# B 线：非连续 KV 整合与 A5 FlashMLA 接线

Status: current — 已完成本地代码整合与静态检查，运行验证未完成；Draft 交付主力机继续开发。

记录版本：2026-09-28 / revision 2。本文件为 PR #12 唯一任务说明，替代 revision 1 的“仅规划”状态及 A3 FlashMLA 扩展提案。

## 1. 当前目标与执行边界

用户已要求发布机先在本地整合两个非连续方案，再参考 1 号迁入我们的 FlashMLA tiling 下沉，以三个功能提交更新 #12，给主力机一个代码起点。**FlashMLA 是 A5 接线**，A3 保留 #16456 的 FIA/component-major 路径。

本轮完成本地实现、静态检查、提交和 Draft PR 更新。没有接管 A 线环境、安装远端依赖、启动服务、发送请求或恢复自动监听。代码整合完成不代表 NPU 验收通过。

## 2. 固定版本与三个功能提交

| 来源 | 完整 SHA | 本次状态 |
| --- | --- | --- |
| vllm-project/vllm-ascend main | `54f350dccd1a64ca89e50d7e900466270a55b2e9` | 已采用，冻结比较基线 |
| vllm-project/vllm-ascend#14340，head 仓库 zhaochuang001/vllm-ascend | `582a4e53dbf830bddee0997ae2c945769f3c05b9` | 第一步完整净增量已整合 |
| vllm-project/vllm-ascend#16456，head 仓库 HackClawMxw/vllm-ascend-fork | `611cf7a70e5d696e8d264c0c38d3d01891c530f5` | 第二步最终布局已整合 |
| 配套 vllm-project/vllm | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | main verified 文件与两 PR 正文所指版本；本组合未实测 |
| 我们现有 Henry-Avery/vllm-ascend#10 | `d7e950dc2a63436fc3c71bd8436533a18925da46` | 第三步接线来源；提取相对旧 base `a583897e0c67d9c23288686728124b04b511a462` 的净差异 |
| 1 号 maoxx241/vllm-ascend-rfc16468-private | `de31c53dc5b94ff246b17aa198404a082162c2f9` | 已核实本地源码的设计参考，不整仓合并 |
| 双机协议 v1.5，Henry-Avery/vllm-ascend#11 | `523de7354af69f4520d40bf365146d45afee480d` | 本会话、子代理与发布准备窗口已读取；主力机回执待补 |

三个功能提交：

1. `5715ae639ad49d4473975ff205b0c9e626409bc5`：#14340 非连续 Attention/Mamba/GQA、state gather/scatter、FLA 消费者及旧算子删除。
2. `51c11ecd3e4aa30f733613d1ef44fa8d683ceaf6`：#16456 MLA single backing、A5 token-fused、A3 component-major、zero/COW 和测试。
3. 本文所属 A5 FlashMLA 功能提交：外部 metadata/main、Prefill FIA / Decode FlashMLA、MRv1/MRv2 图生命周期及接口冲突适配；最终完整 SHA 见 PR 正文，避免提交内自引用。

保留已发布规划提交 `bd85e821`，另有基线同步提交 `ae6eea0e`；它们不属于三个功能阶段。#12 的比较 base 固定为独立分支 `codex/b-line-base-54f350dc`，不移动个人 fork main。

Git 图核对：#14340 与 main 的共同祖先为 `7e2c563f5e6ceddb5b0975753013832d356e096e`；#16456 对应为 `f18d103838b831380d3590df32e0e82d4aa07a9a`。以净变化 squash 整合，保留新版 main，未重放 #16456 中间 BNBD/BBND 反复切换。

## 3. 合并结果与来源边界

两个非连续 PR 均无文本冲突。六个共同修改路径是三个 worker 文件及对应测试：`worker/model_runner_v1.py`、`worker/utils.py`、`worker/v2/attn_utils.py`、`tests/ut/worker/test_model_runner_v1.py`、`test_attn_utils_v2.py`、`test_model_runner_v2_mamba.py`。无文本冲突不等于缓存运行正确。

PR #14340 负责混合物理页、GQA/Mamba view/state 访问与 FLA 迁移；#16456 负责 MLA 单 backing、能力/头数分流及 whole-slot zero/COW。配套 vLLM 的 `AttentionSpec.page_size_bytes` 已包含 `page_size_padded`，未额外叠加 padding。A5 的 `MLA_FLASH` 门控及支持头数来自 #16456，第三步没有给 A3 添加能力。

第三步从 #10 迁移的真实冲突及本次取舍：

| 文件 / 位置 | 决议 |
| --- | --- |
| `worker/v2/attn_utils.py` MLA reshape | 保留 #16456 的 spec.num_heads 分支，舍弃旧补丁中无用的 attn_module 查询；metadata task/scope 单独迁入 |
| `worker/v2/model_runner.py:execute_model` | 保留 main 的 valid_dummy_state_slots、KVPP 和抢占处理，外包 metadata scope；不恢复旧 pcp_dispatch_context/版本分支 |
| 同文件 gather | 保留 main 的 PD decode-recompute 与 uniform-token 处理，再叠加 Flash 阶段排序和 Prefill FULL 图隔离 |
| `worker/v2/spec_decode/dspark/speculator.py` | 保留 main 的 DCPManager、_init_dcp 和 build_attn_metadata_factory，再加入独立 Flash executor/scope |

以上是本次适配决策，不归因于 1 号已有补丁。DSpark/draft 接线保留以避免全局开关下遗漏生命周期，但未宣称运行通过，首轮模型验证仍为普通主模型 no-spec。

## 4. A5 FlashMLA 合同与 1 号参考

| 环节 | 来源、位置与当前决定 |
| --- | --- |
| metadata → main 两段调用 | 参考 1 号 `attention/attention_v1.py:_flash_attention_schedule/_build_flash_attention_metadata`；本候选 `attention/flashmla.py` 沿用 #10 外部 cann_ops_transformer.ops 入口 |
| 稳定缓冲与图外刷新 | 参考 1 号机制；本候选 `attention/flashmla_metadata.py` 和 runner 接线负责 Meta 定容、设备长度刷新、消费等待/复用栅栏；复用已有 executor |
| cache 生产/消费 | #16456 的 MRv1/MRv2 reshape → `attention/mla_v1.py`，传原 A5 PA_BBND backing，保留 stride/offset；writer 使用逻辑 nope/rope view |
| 实参 | #10 外部合同：BF16/FP16，Q TND、KV PA_BBND、output NTD，block128、QK576/V512、max 属性 -1/-1 |
| 路由与输出 | #10 的 Prefill FIA / Decode FlashMLA；短 prompt 尾段仍 Prefill，保留 CPU mirror、writer、V-up、gate、O-proj |
| 与 1 号不同 | 1 号普通路径使用 _C_ascend/PA_BNBD；不照搬 binding/layout/max 属性、全局 Flash Prefill 或 CPU mirror 消除 |

接线范围为 A5 能力门控允许的 local Q heads 8/12/64/96、KV heads1、未量化 KV、PCP=DCP=1、无 KV transfer。代码范围不是二进制支持证据，实际 wheel/OPP 仍须验证。非 A5 开启开关显式失败，关闭时沿用上游布局/计算。没有在 attention 后复制或重排持久 KV。

默认 `VLLM_ASCEND_ENABLE_FLASH_MLA=0`。可选 TRACE 默认关闭；正式性能验证关闭。本次没有搬入 #10 的采样数值诊断器及 SAMPLE_DIAG 变量。

## 5. 静态检查与待验证矩阵

已完成集成差异中 50 份 Python 的 AST、Ruff 0.14.0 check/format，差异空白及任务 Markdown 检查。新增 A3 拒绝/A5 头数门控回归和真实 MRv2 gather 入口的短 Prefill/逐请求排序回归，保留两上游缓存测试和 #10 adapter/metadata 生命周期测试。

未运行 torch/torch_npu UT、构建、单算子、NPU 模型、图 replay、GPQA 或性能。Windows 本地遵守工作区要求不执行 torch 依赖测试。`bash format.sh ci` 因缺少 pre-commit 未完成全仓 hooks。

使用 vllm-ascend-change-validation 对实际 diff 生成了本地计划，run ID `pr12-b-line-integration-20260928`，只有 planned 状态、无运行证据；初始规则为 17 项 required、2 项 recommended。交接矩阵为：

| 顺序 | 必要证据 |
| --- | --- |
| 1 | 配套 vLLM 的 import/build/override，FLA 版本与 ABI、删 binding 后构建 |
| 2 | Flash 关闭的 MLA/GQA/Mamba：非零 offset、padded page、跨页读写、zero/COW、prefix/复用、MRv1/MRv2 |
| 3 | A5 metadata/main：生产 stride/local heads、dtype、零 used/padding、长短长度、独立数值参考 |
| 4 | 普通主模型 eager：纯/短/continued/chunked Prefill、mixed、连续 Decode，Flash 关/开对照 |
| 5 | 同 bucket 改 lengths/table/顺序，多轮 graph replay 对照 eager，确认地址稳定、内容更新与等待 |
| 6 | 授权拓扑的单/多 rank、真实权重、短并发与长上下文，再做固定小集/全量 GPQA |
| 7 | 正确性通过后，同配置比较 metadata kernel、残余同步、TTFT/TPOT/吞吐/HBM |

规则把 chunked-prefill 标为 recommended，本任务因阶段分流改动将其列为交付前必测；性能关键路径也须补性能证据。执行矩阵与预算由后续单独实验单固定，不因没有 NPU 结果降低验收门槛。#10 与 #14340 的历史结果不能替代本候选证据。

## 6. 双机协作任务卡

协议固定入口：[v1.5 简化流程](https://github.com/Henry-Avery/vllm-ascend/blob/523de7354af69f4520d40bf365146d45afee480d/docs/collaboration/simple-day-night-workflow.md)。它是人工流程，不是技术锁或已运行的自动闭环。

| 字段 | 当前值 |
| --- | --- |
| task_id / 主入口 | B-line-noncontiguous-flashmla / PR #12 |
| revision / 模式 | 2 / 本地实现交付；设备执行未开始 |
| 临时实现 owner | 原代码整合窗口，用户明确指定；交付后等待主力机接权回执 |
| 发布准备窗口 | “发布机：PR #12 双机协作准备”，当前 OBSERVE，只读准备 |
| 规范子代理 | #11 只读核对已完成，不持有分支或设备权限 |
| 候选 | PR 正文的完整 head SHA，正式接单时固定，不追随 branch latest |
| env_id / 发布执行 owner | 待明确交接，新窗口不因创建而取得 #10 环境 |
| command_id / run_id / 预算 | 无设备 EXECUTE，无 B 模型 run；后续实验单提供 |
| 仍缺回执 | 主力采用与接权、实际环境/健康状态、有效执行单及发布 ACK |

本轮三个功能提交交付后，原窗口停止主动写入同一实现分支，等待主力接权或用户新指示。发布准备窗口可据固定 SHA 整理验证输入，不能直接部署。

A 线 #10 仍由原 owner 管理，本次未动其 checkout、容器、editable 安装、包、端口、NPU 或自动化。#11 和 HackClawMxw/vllm-ascend-fork#6 未修改。旧回执不证明资源当前空闲，A 旧实验单不成为 B 授权。

正式接单至少固定：协议 SHA、两侧 owner、候选与配套版本、环境卡、control_epoch/command_id、问题与预期、矩阵/预算、停止规则、资源保留/释放责任。跨机送达、主力回执和发布 ACK 尚未取得。
