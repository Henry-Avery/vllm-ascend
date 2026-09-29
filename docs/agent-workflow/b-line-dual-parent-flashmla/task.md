# B 线备选：双父非连续缓存 + A5 FlashMLA

状态：独立代码候选，尚无构建或 NPU 运行证据。本文是备选路线的任务卡；原 [PR #12](https://github.com/Henry-Avery/vllm-ascend/pull/12) 保留为第一版组合及对照，不把两条路线的测试结果互相继承。

## 目标与固定来源

用户要求另起分支，以两个上游非连续方案的实际 head 构成双父合并节点，再对齐固定上游 main，迁入 B 线已有的 A5 FlashMLA Decode/设备 metadata tiling 下沉。首轮模型范围为普通主模型 no-spec；A3 仍使用 FIA/component-major 路径。1 号私仓是两段调用与缓冲生命周期的设计参考，外部算子入口、PA_BBND 和阶段分流以已有 B 线实现为主要迁移来源。

| 角色 | 固定版本 | 本路线中的位置 |
| --- | --- | --- |
| #14340 混合非连续缓存 | vllm-project/vllm-ascend `582a4e53dbf830bddee0997ae2c945769f3c05b9` | 双父合并的第二父提交 |
| #16456 MLA 缓存 | vllm-project/vllm-ascend `52bf27c246e3c9c747267cd1dfce3e63e01020d1` | 双父合并的第一父提交；包含 `611cf7a7` 后新增的测试配置修复 |
| 两个来源的共同祖先 | vllm-project/vllm-ascend `7e2c563f5e6ceddb5b0975753013832d356e096e` | Git 自动三方合并的基点 |
| 对齐的上游 main | vllm-project/vllm-ascend `423e85d032b0146d1d613456d3c391f1f1ac5d13` | 第二次合并的固定主线端点；不是未来移动的 latest |
| 第一版 B 候选 | Henry-Avery/vllm-ascend#12 `eae5347e8f8cd3b22017146ee5789f22942f0311` | A5 功能增量、测试与代码对照；不是本分支祖先 |
| 我们旧接线 | Henry-Avery/vllm-ascend#10 `d7e950dc2a63436fc3c71bd8436533a18925da46` | #12 A5 增量的实际来源 |
| 1 号参考 | maoxx241/vllm-ascend-rfc16468-private `de31c53dc5b94ff246b17aa198404a082162c2f9` | 两段调用、稳定缓冲与图外刷新参考；不整仓合入 |
| 拟议配套 vLLM | vllm-project/vllm `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | 当前只固定版本，组合未经安装或运行验证 |
| 双机协议 | Henry-Avery/vllm-ascend#11 文档 `523de7354af69f4520d40bf365146d45afee480d` | 继续采用 v1.5；新候选必须另行登记接单与结果 |

双父合并提交 `485b0c691384b665d20c45651d47d468b247fee9` 由两个固定来源 head 生成，无文本冲突。其后把固定 main `423e85d` 合入的提交为 `bccc3c9c3a4a9e5aaf0775312d9298ae8d40b799`。主线合并在 `tests/ut/worker/test_attn_utils_v2.py` 有一处相邻测试冲突，保留了上游的 draft per-request CPU upper-bound 测试和来源方案的 physical-row 测试。两次 Git 合并成功只证明文本层可整合。

## A5 合同与迁移边界

从 #12 的第三个功能提交迁入外部 `cann_ops_transformer.ops` metadata/main 两段调用、稳定设备缓冲、MRv1/MRv2 图生命周期、Prefill FIA / Decode FlashMLA 阶段路由及测试。保留新版 main 的长度上界、KVPP、PD recompute 和 DCP 处理。实际缓存 backing、view stride、storage offset、zero/COW、设备长度、请求排序、图回放地址和复用等待必须在新基线上再次核对。

接线声明的 A5 local Q heads 8/12/64/96、BF16/FP16、QK576/V512、block128、未量化 KV、PCP=DCP=1、无 KV transfer 来自 #12 的适配范围；真实 wheel/OPP 的 stride、头数和 ABI 支持尚未验证。DSpark/draft 接线若保留，只表示避免生命周期缺口，不表示其运行通过。

## 协作与验收

此分支与 #12 分离，原 B 线候选和证据保持各自固定 SHA。代码候选由本窗口在隔离本地 checkout 整理；主力机后续接权、发布执行 owner、B 环境卡、有效 EXECUTE 实验单与预算仍待明确。本轮不触碰 A 线 #10 的容器、editable 安装、端口或设备，不因新分支自动部署。

交付前至少完成：基于真实 diff 的来源/冲突核对；静态、构建与必要 UT；Flash 关闭的 MLA/GQA/Mamba 缓存基础；A5 实际 stride 单算子；普通主模型 eager 的 Prefill/continued/chunked/mixed/Decode；动态输入多轮 graph replay；获准拓扑的模型精度；正确性通过后的性能。每项运行结果绑定最终完整 SHA、配套软件、实际环境、命令、参数、run_id、原始日志和未覆盖范围。#12、上游作者或参考仓的历史结果不自动计入本候选。
