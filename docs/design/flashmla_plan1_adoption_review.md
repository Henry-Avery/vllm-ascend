# 1 号方案借鉴复核与 #14 检查点

## 比较对象与结论

本次固定比较本地 1 号 `de31c53dc5b94ff246b17aa198404a082162c2f9`、
旧 2 号 `a583897e0c67d9c23288686728124b04b511a462`、
共享运行候选 `d7e950dc2a63436fc3c71bd8436533a18925da46`。
PR #14 首个诊断提交为 `b0b0d5f1678782163df1a46b87aa004fe6fdbd12`；本次在其后补充取证。
这不是新主线迁移，也未将 #16456 后来的提交混入旧候选。

[另一个仓库 PR #6](https://github.com/HackClawMxw/vllm-ascend-fork/pull/6)
仍指向相同 d7，其移动后的 base 不适合直接当旧基线增量比较。
它提供历史审查和运行线索，不是新的修复。其旧“启动失败”摘要也不能覆盖
[后来 d7 的 C1 回执](https://github.com/Henry-Avery/vllm-ascend/pull/10#issuecomment-5870002634)。
该回执报告同 payload、同 DP、同参数可失败也可成功，且 C1 失败时已超采集预算；
不能将更早 C60 的 raw-logits NaN 结论直接套到这些未采集的失败上。
没有原始设备张量在本地，以上按发布机回执理解。

**不能说“1 号能跑，所以整套搬过来一定修复”，也不能说“接口不同，所以其中的逻辑都不能借”。**
整体算子替换有真实接口限制；computed-prefill 来源、非空历史筛选与 LSE 规范化可以局部借鉴。
后几项暂未改运行逻辑的原因是缺少对应失败证据与适配回归，不是已有证据证明旧写法更好。

## 为什么没有直接使用 1 号

路径均相对于对应固定版本的仓库；函数名比随提交移动的行号更适合定位。

| 项目 | 1 号做法 | 当前旧 B 线做法 | 未照搬的理由及性质 | 下一步判据 |
| --- | --- | --- | --- | --- |
| 整体路由 | Flash 开关进入 `_forward_flash`，可走专用 full-prefill | Prefill FIA、Decode external FlashMLA | **方案边界**：整体替换会同时更换 Prefill 算子；不是已证实的正确性优势 | 先确认 1 号实际命中路由；不要仅凭文件里存在 full-prefill 就断言运行过 |
| Prefill 算子接口 | `native_flash_adapters` 的 192/128 非吸收注意力，探测 metadata schema 的 `head_dim_v` | 公共 external MLA Decode 接口，latent512+rope64、value512 | **直拷有接口约束**：现有 Decode 适配器固定 576/512，不能直接代替 192/128 Prefill；发布机包能力仍需核实 | 读实际包/schema/维度；1 号还有 absorbed fallback，不能把全套 Flash Prefill 都说成不支持 |
| 历史长度 | `_build_full_flash_prefill` 直接消费 computed-prefill | FIA builder 用 `seq-query` | **可借鉴，未证明必须改**：本次旧 a583→d7 AST 比较中 builder 原样保留；无 spec 的 CPU 字段存在更新链 | 对同请求 computed-prefill、CPU/device seq、query、positions；若不等，先找更新源/轮次，再改权威字段 |
| 零历史行 | `build_prefill_plan` 用 `length > start` 选 active；重排 query/page table | FIA 每个历史分段保留全部 prefill 行，允许 KV 长度为零 | **可借鉴，无硬阻碍**：需要同步改 query/页表/累积长度/写回映射；尚未证实当前算子空行违规 | 空行 LSE 应为 `-inf`；输出可未定义。比较实际 merge 是否忽略其贡献；命中偏差后移植 active-row 策略 |
| 合并顺序与 LSE | 规范化 LSE，逐段合并有效行，index-copy 回原位置 | 当前块和历史块以旧 FIA 路径合并 | **可借鉴，需要适配**：轴布局与 request 行映射必须一致；顺序不同可产生舍入差异 | 同时比较“CPU 理想分支”和“实际 FIA 分支的 CPU merge”，区别分支错误与 merge 错误 |
| KV 布局/写读 | 多路 native cache 合约，还含量化/并行扩展 | 复用 2 号 fused BBND 及 latent/rope 视图 | **接口与任务范围**：不能连 allocator/layout 一起盲搬；2 号 fused allocator 不是本次新增 | 实际 stride/offset、slot、跨页 gather 精确比；已有单页 NPU probe 通过不覆盖长请求和复用 |
| Query/投影 | 各 Flash 分支自己的 Q、吸收与投影衔接 | 复用旧 `_q_proj_and_k_up_proj`、`_v_up_proj`，Decode 外部返回 NTD 后接旧投影 | **局部接线选择**，不是证明一致 | 所选层 CPU attention + 投影前后快照；全 MLA 边界与 model/LM-head 检查缩小范围；尚未独立重算所有投影 |
| CPU length 去同步 | `_use_device_seq_lens` 受 K3、PCP 与 executor 等条件限制 | 保留 MRv2 speculative D2H 与 event wait；FIA 仍有 CPU 消费者 | **有消费者约束**：去同步是单独性能工程，不能只搬开关 | 当前无 spec 故障优先级较低；扩展 spec 前审计所有 CPU 消费者与 rejection |
| 图参数更新 | Flash 开启时全局跳过 MLA `update_graph_params` | 过滤 external Flash 层，保留 FIA 层更新 | **混合后端的必要适配**：全局跳过可能漏 FIA | 另做 target/draft capture/replay 证据；当前 eager 不能验收图，也不优先归咎图更新 |
| metadata 稳定地址 | 为图使用稳定 buffer 与图外更新 | 独立 schedule 刷新、executor submit/wait/release | **已借鉴但未证明无竞态**：源码等待链只覆盖接入任务 | 同轮 inputs 数值/身份对照；同步诊断通过不能排除无同步复用竞态 |
| eager buffer 保留 | 图稳定地址策略 | 曾按 eager 形状永久缓存，`ae92cb43e` 已改为仅显式图容量保留 | **已确认并修复的接入问题**，d7/#14 已含 | 不再重复当待修问题；保留 builder 生命周期回归 |
| C8、DCP/PCP 等扩展 | 1 号包含额外实现/路由 | external 首轮限非量化、PCP=DCP=1 | **范围限制**，不是当前受支持路径错误的解释 | 当前复现不靠扩大支持面解决 |

1 号对应代码：`attention/mla_v1.py`、`attention/mla_prefill.py`、`worker/v2/model_runner.py`。
B 线新增接线集中在 `attention/flashmla.py`、`attention/flashmla_metadata.py`、
`attention/mla_v1.py`、`worker/v2/attn_utils.py`、`worker/v2/model_runner.py` 和图/DSpark 路径。

## 实际动过什么，优先检查什么

旧 a583→d7 的 AST 对照确认以下六个函数计算主体相同：
`build_chunked_metadata`、`build_prefill_metadata`、`_compute_prefill_context`、
`_forward_prefill`、`_q_proj_and_k_up_proj`、`_v_up_proj`。
这排除了“本次直接改坏了这些函数内算术”这一说法；它们的新输入、路由和 buffer 生命周期仍可能出错。

1. **输入接线与身份**：MRv2 phase 重排后的 request、query、seq、idx_mapping、page table、positions 是否一致；CPU computed-prefill 是否对应同轮。
2. **共享 KV 写读**：Prefill 写入供后续 Flash decode 使用；错误页/slot、历史污染、视图 stride/offset 均可在新消费者处暴露。
3. **当前/历史/merge**：保留代码不等于在新输入下无错；重点是非空历史、短尾、零行与 LSE。真实单请求首块全零历史通常直接跳过历史分支，后续单请求又有正历史；若确认内部一直单请求，混合零行假设优先级应下降。
4. **外部 Decode**：刷新 lengths/cu/used_q/table/slot/position、主算子输出、NTD 接回投影；以实际页表参考做数值比较。
5. **非 attention 或后段层**：只观察头两层不能排除后续 attention、MLP/MoE、norm 或 LM-head。新增边界将异常夹在更短的区间，但不重算这些模块。
6. **时序/复用**：相同输入间歇失败使状态污染、未初始化或复用值得查；它不是竞态已经成立的证据。同步取证可能改变复现率。

## #14 本次补充及判读

| 新增检查 | 填补的缺口 | 结果的限制 |
| --- | --- | --- |
| `request_ids` 门控整个 attention/sampling 窗口 | 原先只筛层内样本，无关 smoke 仍耗完 128 步 | 必须是内部精确 ID；未命中明确报缺失。不是自动识别失败请求 |
| `start_after_forwards` | 可跳过已知的前期真实 forward，profile/dummy 不计 | 各 worker 独立计数，不等于 HTTP 请求数；开始前的输出未观察 |
| `scan_all_mla_layers` | 参考仍限少数层，但逐层扫描所选请求首尾 query 的 MLA 输入/输出 | 默认关闭；推荐本轮开启。仅检查有限性，不判断有限但错误的值；SP 局部行不能映射时记 skipped |
| 首个非有限 MLA 边界快照 | 后段层出 NaN 不再只有 sampling 终点信息 | 每 forward 最多一个坏边界快照，受已有保存预算限制 |
| model output → 实际 LM-head input | 区分模型已坏、hidden 行选择错误与进入 logits 后才异常 | 只检查真实 logits 对应行，忽略 padding；没有重算 LM-head 权重/通信 |
| 验收器核对上述覆盖并输出 `first_observed_anomaly` | 缺边界或缺目标不能误报通过 | 按同 worker 记录时间找首个已观察异常；不做跨机器因果推断，不等于根因 |

排查时先看首个异常所在 forward/request 及 Prefill/Decode 身份，再看对应层/阶段。
若 MLA 输入有限、输出不有限，缩到该 attention 层；若前一 MLA 输出有限、下一层输入不有限，
查中间 residual/MLP/MoE/norm。若 model output 正常而 LM-head 输入映射不一致，查行选择；
若两者均正常而 raw logits 已坏，再查 LM-head/通信。有限值仍需要数值参考，不能仅靠上述分界验收精度。

生产 Prefill/Decode、空行处理和 merge 算法此次不变；修改的是关闭时不执行的诊断接线。
[采集配置、预算和运行回执要求](flashmla_chunk_diagnostics.md)给出完整操作方式。
