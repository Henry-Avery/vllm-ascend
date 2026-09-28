# B 线：非连续 KV 方案整合与 FlashMLA tiling 下沉计划

Status: current — 仅规划，尚无 B 线运行候选或运行验证。

计划版本：2026-09-28 / revision 1。

## 1. 目标与本轮授权

用户将 B 线交给本发布机上的新会话：先把新版上游 #16456 与 #14340 整合到同一代码基线，有冲突先处理冲突，再迁入我们的 FlashMLA 算子接入及 tiling 下沉；实现参考仍为本地 1 号私仓。当前明确要求先规划，并把计划发布到新的 Draft PR。

“合到一个节点”本计划解释为一个可固定 SHA、可部署的集成代码节点，不据此把原模型的分布式拓扑缩成单机。实际模型、拓扑和 NPU 资源在执行单中固定。

本轮交付是本文和独立 Draft PR。实现、冲突修复、运行部署和自动循环属于后续阶段；本文没有把这些阶段标为已完成。后续实施由本会话承接 B 线，用户这次指定的分工优先于协议中发布机默认仅做验证的角色约定。

## 2. 与 A 线及双机协议的关系

采用 [协作协议 v1.5 固定入口](https://github.com/Henry-Avery/vllm-ascend/blob/523de7354af69f4520d40bf365146d45afee480d/docs/collaboration/simple-day-night-workflow.md)，固定文档提交为 `523de7354af69f4520d40bf365146d45afee480d`。主力/发布是机器角色，A/B 是业务工作线，两者不能混用。

| 项目 | 本计划约定 |
| --- | --- |
| B 线计划与后续本地实现 owner | 发起本 PR 的发布机 B 线会话，用户已指定承接 |
| A 线控制入口 | [Henry-Avery/vllm-ascend#10](https://github.com/Henry-Avery/vllm-ascend/pull/10)，保留旧基线排障与证据 |
| B 线控制入口 | 本计划所属的新 Draft PR；当前文档为唯一计划，后续候选及结果引用固定 SHA/run ID |
| A 线在途状态 | 本轮只读任务记录显示已有发布执行会话正在进行 HCCL 双机启动排查；尚不能据此宣布环境恢复 |
| 共享运行环境 owner | 继续由现有发布执行会话持有；本会话没有接管运行容器、依赖、端口或 NPU |
| B 线设备执行 | 后续交给唯一发布执行 owner 串行执行，或完成明确交接后再接手；独立目录不代表隔离 editable install 和设备 |
| 当前后台自动化 | 本轮未检查、修改或恢复；不声称两端已自动采用、已互通或已恢复闭环 |

协议可以作为人工协作流程使用，但它只是版本化规则，没有实现技术锁、调度器或跨机暂停。当前仅确认本会话已读取固定版本；另一端采用、消息送达及执行接单仍须实际回执。文档发布本身不能证明整个双机流程已经跑通。

隔离要求：

- B 使用独立 checkout 和 `codex/b-line-*` 分支；不切换、覆盖 A 的 checkout 或共享 editable import。
- 不推进 #10 的实现分支、不改 #11、不处理 HackClawMxw/vllm-ascend-fork#6 的移动 base 冲突。
- 不把 A 的旧实验单重放成 B 实验；A 的暂停或恢复状态由 A 当前有效指令管理。本次新授权只覆盖 B 计划与 PR。
- 后续每次设备执行登记候选、配套 vLLM、环境差异、owner、run ID、测试问题、预算和停止规则。
- 本轮不创建新监听器，不向其他会话发送执行消息，不启动子代理。

## 3. 固定输入与来源

以下为 2026-09-28 本轮读取的版本。观察到不等于已采用，源码检查不等于运行验证。

| 角色 | 仓库 / 引用 | 固定 SHA | 状态 |
| --- | --- | --- | --- |
| 上游/main，拟议实现起点 | vllm-project/vllm-ascend main | `54f350dccd1a64ca89e50d7e900466270a55b2e9` | 已解析；未集成 |
| 上游方案 2 | [vllm-project/vllm-ascend#16456](https://github.com/vllm-project/vllm-ascend/pull/16456)，head 仓库 HackClawMxw/vllm-ascend-fork，分支 0913main | `611cf7a70e5d696e8d264c0c38d3d01891c530f5` | 已读相关布局补丁；不是旧 `4a791167` 或 `a583897e` |
| 上游混合非连续方案 | [vllm-project/vllm-ascend#14340](https://github.com/vllm-project/vllm-ascend/pull/14340)，head 仓库 zhaochuang001/vllm-ascend | `582a4e53dbf830bddee0997ae2c945769f3c05b9` | 已读文件清单及相关分配/reshape 补丁 |
| 拟议配套 vLLM | vllm-project/vllm | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | 两 PR 正文及上述 main 的 `.github/vllm-main-verified.commit` 指向此版本；组合未实测 |
| 我们现有候选 | Henry-Avery/vllm-ascend#10 | `d7e950dc2a63436fc3c71bd8436533a18925da46` | 迁移来源；旧 base 为 `a583897e0c67d9c23288686728124b04b511a462` |
| 1 号命名参考 | maoxx241/vllm-ascend-rfc16468-private | `de31c53dc5b94ff246b17aa198404a082162c2f9` | 已核实本地 HEAD、remote 及 metadata/MLA/runner 代码；私仓不是公共依赖 |
| 本文 PR 的文档 base | Henry-Avery/vllm-ascend main | `aff1b74b66467a7805cde69ef0728b7e32c0f990` | 仅承载计划文档，不是 B 集成运行基线 |

PR #16456 API 当时报告 base `597ab74d86af0e5d9a467618d70bd0aafed0cbea`；#14340 报告 base `8d4409d6256d8a6729140ddcc0d1889e3f96cdd6`。它们是读取到的 PR base 信息，不当成两者共同基点。实施时以本地完整 Git 图计算共同祖先、提交关系和净差异；不盲目重放分支上全部主线提交。

本 PR 以个人 fork main 为目标，仅新增本文，因此不会混入运行代码。后续实现使用单独集成分支；如需运行代码 PR，先固定其比较基线和差异，再从本控制入口关联，不能把当前文档 base 冒充新版 main。

## 4. 已确认的重叠与语义风险

两个上游 PR 同时修改以下六个路径。它们是文件级重叠清单，尚未执行 merge，不能声称每个文件都已发生文本冲突：

- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/worker/utils.py`
- `vllm_ascend/worker/v2/attn_utils.py`
- `tests/ut/worker/test_model_runner_v1.py`
- `tests/ut/worker/test_attn_utils_v2.py`
- `tests/ut/worker/test_model_runner_v2_mamba.py`

| 来源及代码位置 | 已有行为 / 风险 | B 线处理计划 |
| --- | --- | --- |
| #14340 `582a4e53`：`patch/platform/patch_mamba_config.py`、`worker/v2/attn_utils.py` | Attention/Mamba 逻辑页大小可不同，按物理 padded page 建 view；保留 legacy MLA 分支 | 先建立公共物理页、逻辑页、offset 和块计数合同，再接 MLA 专用分支 |
| #16456 `611cf7a7`：`core/kv_cache_interface.py`、MRv1 `_uses_single_raw_mla_cache`、MRv2 同名 helper / `_reshape_kv_cache_v2` | 合适的 MLA 使用 single raw backing；根据硬件能力和 local Q heads 选择 token-fused 或 component-major | 保留类型与能力门控，不能用统一 `ours/theirs` 覆盖整个 allocator |
| #16456：`device/hardware_profile.py`、两个 runner 的 reshape | A3/FIA 默认 component-major 双 view；支持 `MLA_FLASH` 且头数合适时才是 token-fused Tensor | **新增适配提案**：外部 FlashMLA 在分配/reshape 前明确申请其 fused 合同，开关关闭保留上游默认；具体设计待消费链检查，不全局伪造硬件能力 |
| #16456：`worker/utils.py`、`worker/v2/utils.py` | whole-slot zero/COW 与 fused backing 生命周期相连 | 与 #14340 的 padded-page zero、块映射同时核对，覆盖完整物理 slot，避免只清/拷逻辑 slice |
| #16456：`attention/mla_v1.py` | fused Tensor 在原 MLA 中零拷贝拆为 nope/rope，并避开不合适的 fused prolog | Flash 接口保留原 backing 身份，writer 和原 FIA Prefill 必须能消费所选 view；不能在 attention 后重排持久 KV |
| #14340：`ops/gdn.py`、`ops/kimi_kda.py`、`ops/triton/mamba/state_index.py`、`csrc` 与 binding | PR 包含 gather/scatter、GDN/FLA 调用变化及旧算子删除，不只是 stride 修改 | 对整个 PR 做依赖闭包审查，核对 FLA 实际版本/API、构建和 K3 KDA 消费者；不能只挑 allocator 而遗漏必要消费者，也不自行恢复已删除算子 |

公共合同至少明确：raw backing 所有者与共享关系、logical/physical page bytes、manager/kernel block 换算、storage offset 单位、每轴 stride、nope/rope 排布、zero/COW 覆盖范围、MRv1/MRv2 差异和不支持分支的回退/报错规则。重点核实 `page_size_bytes` 与 `page_size_padded` 在两个方案中的最终含义，避免 padding 计算两次或漏算。

参考代码位置：[新版 MLA reshape](https://github.com/HackClawMxw/vllm-ascend-fork/blob/611cf7a70e5d696e8d264c0c38d3d01891c530f5/vllm_ascend/worker/v2/attn_utils.py)、[混合非连续 reshape](https://github.com/zhaochuang001/vllm-ascend/blob/582a4e53dbf830bddee0997ae2c945769f3c05b9/vllm_ascend/worker/v2/attn_utils.py)、[state gather/scatter](https://github.com/zhaochuang001/vllm-ascend/blob/582a4e53dbf830bddee0997ae2c945769f3c05b9/vllm_ascend/ops/triton/mamba/state_index.py)。本节仅为源码观察与风险分析。

## 5. 实施顺序与里程碑

### M0：冻结代码与依赖

在 B 独立实现 checkout 获取固定 Git 对象，审计 main、两个 PR 的共同祖先和端点净差异。以已固定新版 main 为起点，优先整合 #14340 的通用混合框架，再叠加 #16456 的 MLA 专用布局；这是本计划提出的顺序，不是上游承诺的堆叠关系。如果 Git 图证明一方已包含所需补丁，就去重而不重复应用。

同时固定 vLLM、FLA、torch/torch_npu、CANN、外部 FlashMLA wheel/custom OPP、镜像 digest。FLA 与外部算子具体版本目前待核实；A 的旧 vLLM `84030bbe` 环境不能直接宣称满足 B 的 `ced6857`。

产物：固定引用表、依赖闭包、真实冲突清单。输入 head 以后变化时另记观察版本，不静默替换本轮采用版本。

### M1：只整合非连续缓存基础

先处理公共 allocator、spec、reshape、zero/COW 和消费者分支，再整合测试。对每一处冲突记录“两个来源分别想保留什么、最终如何兼容、由什么用例证明”。保留 #14340 的 GQA/Mamba 正确性，同时保留 #16456 的 MLA 单 backing、两种排布和既有排除条件。

提交按依赖整合、语义冲突修复及回归分层，得到第一个固定集成 SHA。FlashMLA 开关保持关闭，先证明缓存基础没有退化。M1 成功才进入 M2；不能用 Flash 输出掩盖基础布局错误。

### M2：迁入 FlashMLA 算子接线与 tiling 下沉

从 #10 相对旧 base 的功能差异逐项迁入，禁止整支 merge 把旧 runner/allocator 带回新版基础。按以下来源界限实施：

| 内容 | 来源与适配决定 |
| --- | --- |
| 设备 metadata 算子生成 schedule，再由 FlashMLA 主算子消费 | 借鉴 1 号 `de31c53d` 的 `attention/attention_v1.py:_flash_attention_schedule`、`_build_flash_attention_metadata` 及 `attention/mla_v1.py` |
| Meta 定容、稳定缓冲、图外刷新与消费/复用等待 | 借鉴 1 号机制；对照集成基线已有 `worker/device_metadata.py`，仅迁移缺失接线，不把已有 executor 误记为新增 |
| `cann_ops_transformer.ops` 外部入口、PA_BBND、max 属性为 -1 | 我们 #10 的适配；1 号普通路径是 `_C_ascend` binding 和 PA_BNBD，不能原样复制实参 |
| Prefill FIA / Decode FlashMLA、真实阶段分流、保留 FIA CPU length 消费 | 我们 #10 的既有策略；不照搬 1 号全局 Flash forward 或特定 K3 CPU mirror 消除 |
| 新版 A3 component-major 与外部 Flash fused 合同衔接 | 本次 B 线新增设计，尚未实现；在 cache 生产端解决，禁止依赖 attention 后复制整份 KV |
| A 线采样数值诊断模块 | 可复用取证工具，单列可选提交；不是完成 tiling 下沉的必要运行补丁，不默认开启 |

主要迁移位置是 `attention/flashmla.py`、`attention/flashmla_metadata.py`、`attention/mla_v1.py`、`envs.py` 和 MRv1/MRv2 的 metadata/capture/replay 接线。DSpark/draft 修改逐项分离；本阶段先验证普通主模型 no-spec，后续支持项必须追加明确验证。

维持 #10 已约定的初始路径：未量化 BF16/FP16 MLA KV、block128、latent512+positional64、Q TND、KV PA_BBND、output NTD、PCP=DCP=1、无 KV transfer。头数按 TP 后 local heads 核对，#10 接线声明的 8/12/64/96 仍须与实际 wheel、硬件和新版 layout selector 逐项匹配，不能仅凭源码常量宣称均已支持。

设备长度、cu/used_q、页表与 schedule 必须来自同一批次；capture/replay 中地址稳定、内容更新、padding 不写缓存、上一轮消费完成后才复用。保留原 writer、V-up、gate、O-proj 语义。接入 tiling 下沉不等于消除 runner 的所有 CPU 同步。

### M3：固定候选后分层验证

最终集成 diff 形成后使用工作区 `vllm-ascend-change-validation` 生成正式验证计划，其他执行技能产出关联 Run Manifest。本轮尚无集成 diff，因此下面是人工预规划，不伪造 planner 或测试结果。

| 层次 | 最小验收内容 | 版本比较 |
| --- | --- | --- |
| 静态/构建/UT | 六个重叠路径的回归；依赖导入及删算子后的 binding/build；错误配置与 legacy 分支 | 固定 main、M1 |
| 缓存生命周期 | dense/padded、非零 offset、两种 MLA 排布、GQA/Mamba state、manager/kernel 块换算、零页、跨页 COW、padding/邻层保护、MRv1/MRv2 | M1、Flash 关闭 |
| 单算子真实 NPU | 使用生产 stride 和 local heads；metadata 与 main 合同、零 used 行、短/跨页/长长度、BF16/FP16、输出及 LSE 对可信参考 | M2 固定 SHA/真实 wheel |
| 模型 eager | 纯 Prefill、短 continued Prefill、mixed、连续 Decode、多请求复用；确认 Prefill FIA / Decode Flash 路径；覆盖 K3 及 #14340 影响的代表性 GQA/Mamba | 同一 M2，Flash 关/开；M1 作为缓存回归对照 |
| 图 replay | 同一 bucket 改长度、页表、请求顺序，连续多次 replay；metadata 地址/刷新/等待及 padding；与 eager 数值/输出比较 | 同一 M2 eager/graph |
| 分布式与精度 | 环境健康后再做既定 K3 拓扑、短并发、长上下文、固定小集，再决定全量 GPQA；保持样本、采样、token 上限和评分口径一致 | 每轮固定候选和配置；不继承 #10 历史分数 |
| 性能 | 正确性通过后，同环境同负载比较 Flash 开关及必要的 M1/M2；记录 metadata kernel、残余同步、TTFT/TPOT/吞吐和 HBM | 有预热与重复测量的可比运行 |

Windows 本地不运行 torch_npu 测试。远端执行沿用用户指定工作区的工具与本地技能，NPU 管理按当次有效执行约定办理；本文不授权改造 A 的运行方式或抢占设备。

PR #14340 作者报告的 Qwen/GQA 与 GPQA 结果仅属于其报告配置；#10 的旧 C8 通过、高并发异常及 d7 启动失败也仅属于各自 SHA。它们均不能替代 B 组合验证。PD、量化 KV、DCP 扩展和把 FlashMLA 用于 KDA 计算本身不纳入首个里程碑；但 #14340 触及的 KDA/state 现有行为必须做受影响回归，不能以此为由忽略。

### M4：交付与状态判定

交付包含固定 B 候选及配套 SHA、逐项来源/冲突决议、实际环境清单、最小复现命令、原始证据、已通过/失败/未覆盖矩阵。缺少必要 NPU 或图证据时保持 Draft，明确验证不完整；不以 merge 无文本冲突、服务启动或测试脚本 exit 0 代替模型验收。

## 6. 当前状态与下一步

- 已完成：读取 v1.5 固定文档、核对 A 在途任务、固定两个上游 head 与 1 号本地参考、识别六个重叠文件和 A3 布局差异、整理本计划。
- 尚未完成：实际合并与冲突探测、B 实现、FLA/算子包 ABI 核对、双端采用回执、B 环境接单以及任何 B 运行测试。
- 下一步：用户进入实施阶段后从 M0/M1 开始，先交付“两个非连续方案的固定集成 SHA + 冲突决议 + 缓存基础证据”，再进行 M2。A 线可由原 owner 继续其有效授权范围内的工作，不因本计划换码或重启。

本轮文档检查与提交 SHA 记录在 PR 正文。本文是规划与来源记录，不是 EXECUTE 实验单，也不是“两套方案已经合好”或“环境已可用”的证明。
