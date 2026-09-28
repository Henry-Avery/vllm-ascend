# FlashMLA chunk、KV 与合并诊断

此候选基于 PR #10 的旧验证线 `d7e950dc2a63436fc3c71bd8436533a18925da46`，
补充 sampling 诊断之前的 attention 证据。它没有修改 chunk 算法，
也不表示连续 `!` 已修复。CPU 回归验证诊断逻辑；真实 NPU 结果仍待人工启动验证。
由用户安排发布机空闲时验证；本次推送不自动启动服务或发起负载。

## 要区分的三个问题

关闭 prefix caching 不会消除同一请求前几轮 prefill 写入的历史 KV。
例如本轮 query=3、seq=259，历史应是 `[0, 256)`，当前块是 `[256, 259)`。
`seq-query` 与 computed-prefill 正常应相等；单凭采用 CPU 镜像不能判定过期。
诊断在 prepare_inputs 后冻结请求顺序、query 边界、computed-prefill、CPU 长度及状态索引，
在实际层执行时对照设备值、位置、slot 和页表。

本地参考实现的 `full_prefill` 分支也分当前块和历史块，并按 LSE 合并。
它筛掉每段没有历史的请求，重新选择 query/页表，逐段合并后写回有效行。
PR #10 的 FIA 路径保留零历史行，最后统一合并所有分段。
参考路径还受算子包接口、维度及并行配置约束，必须有运行路由证据才能断言曾命中它。
这种差别是排查线索，尚不是已确认根因。

空历史 `LSE=-inf` 可以合法，其输出可能未定义；CPU 参考将其贡献显式置零，
避免 `0*NaN` 污染。非空输出即使全有限，也必须与参考比较。
同一个 forward_id 关联的是观测调用，不是对所有缓冲区生命周期的形式化证明。
同步复制可能改变调度或掩盖复用竞态，诊断开启后不再复现不代表修复。

## 采集内容

| 检查 | 实际对照 |
| --- | --- |
| 请求和本轮身份 | forward_id、层、rank、request ID、CPU/设备请求索引、query 边界、seq 和 positions |
| 历史范围 | `seq-query` 对 computed-prefill，非空分段连续且总长度等于历史，FIA 累积长度与 gather 页表 |
| KV writer | 当前 prefill 的采样位置推导 slot，源 latent/positional 与写后 paged cache 精确比较 |
| KV gather | 按真实页表独立跨页索引，完整比较选中请求该历史段的 latent/positional；记录 stride/storage offset |
| FIA | 选中 query 行、全部本地 heads 的 CPU FP32 causal/current 与 noncausal/history 参考，比较 output/LSE |
| merge | 分别对照 CPU 理想分支与实际 FIA 分支的 merge；保存空历史实际输出，并检查零历史请求合并后等于当前块输出 |
| Flash decode | 刷新后的长度、cu、used_q、页表、slot、position 对照本轮 metadata；latent 输出对照 paged KV 参考 |
| 全层边界 / LM-head | 可选全 MLA 输入/输出有限性、首个坏边界快照；model output 与 LM-head 实际输入逐请求比较 |
| projection / sampling | 采样行的投影前后快照与有限性，沿 forward_id 关联既有 raw/processed logits 和最终 token |

writer 目前直接对照 prefill 写入；decode 提供 metadata 和 attention 结果检查，
没有独立重算 decode writer 源值。投影只记录及检查有限性，没有重算权重矩阵乘法。
这两项不能报告为完整算子正确性证明。

## 发布机拉取候选

候选分支是 `codex/flashmla-chunk-diagnostics`，PR 以旧验证分支
`codex/flashmla-tiling-oldmain-v2` 为 base，增加诊断提交。
它不包含新主线 `54f350dc…` 的迁移；旧 PR #10 分支保持原状。
PR 说明记录候选完整 SHA，发布机应固定该 SHA，不以未来移动的分支头代替实验版本。

在干净的验证 checkout、确认 origin 指向 Henry-Avery/vllm-ascend 后执行：

```bash
git fetch origin codex/flashmla-chunk-diagnostics
git switch --detach FETCH_HEAD
git rev-parse HEAD
git merge-base HEAD origin/codex/flashmla-tiling-oldmain-v2
```

HEAD 应等于本 PR 记录的候选 SHA，merge-base 应为
`d7e950dc2a63436fc3c71bd8436533a18925da46`。
继续使用匹配的 pinned vLLM `84030bbe3d74d99bad477a3d2e37a973ccd8865c`，
保留原算子包和模型配置；核对实际导入的 vllm_ascend 路径来自该 checkout。

## 开启方式

在获准的原 eager 启动配置中，保留原模型、算子包、采样参数和失败 payload。
worker 必须继承以下变量；每轮使用全新的绝对输出目录：

```bash
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR=/absolute/path/to/run/diagnostics
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_STEPS=128
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_ROWS=2
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DP_RANK=-1
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TP_RANK=0
export VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TOKEN_IDS=0
export VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG=1
export VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG_CONFIG='{"max_layers":2,"requests_per_phase":2,"query_rows":2,"max_kv_tokens":4096,"max_saved_mib":128,"scan_all_mla_layers":true}'
```

需要已有 `VLLM_ASCEND_ENABLE_FLASH_MLA=1` 和 enforce_eager，禁止 speculative decoding。
sampling 开关既有的标准 MRv2 sampler、PP=1 等限制继续适用；外部 FlashMLA 已限制
PCP=DCP=1、非量化 BF16/FP16 BBND cache。chunk 开关默认为 0；单独开启却没有 sampling 目录会报错。
TOKEN_IDS=0 仅是候选监视 ID，仍需用实际 tokenizer 确认 `!` 的编码。

默认观察最先遇到的两个 attention 层，每层每轮分别最多两个 prefill 和两个 decode 请求，
每请求两个 query 位置（首尾）。有历史和零历史 prefill 共存时优先各取一个。
默认覆盖所有 DP 的 TP0；其他 TP 没有被验证。推荐配置另外开启全部 MLA 层的有限性扫描，但只观察选中请求的首尾 query；
它不提供其他层的完整数值参考。
可用 `layer_names` 选参考层，`request_ids` 选内部精确 request ID，仍受数量限制。
`request_ids` 现同时门控 attention 和 sampling，未匹配的 forward 不消耗 128 步预算。
不知道内部 ID 时留空，并在新进程的诊断窗口内先跑目标 payload。
若指定 ID 只发到一个 DP，DP 选择和验收预期列表也应对应该副本；其他副本未命中不能算通过。

JSON 参数范围：max_layers 1–8，requests_per_phase 1–8，query_rows 1–4，
max_kv_tokens 1–16384，max_saved_mib 1–1024；layer_names 最多 8 项，request_ids 最多 16 项。
`start_after_forwards` 默认 0，范围 0–1000000，各 worker 独立跳过指定数量真实 forward；
`scan_all_mla_layers` 默认 false，只接受 JSON boolean，推荐本轮 true。
atol/rtol 默认均为 0.05，范围有限的 0–1；这是排查容差，不是模型精度验收标准。
metadata、writer 和 gather 使用精确比较。

采集窗口继承 SAMPLE_DIAG_STEPS（默认 64，最多 128 个匹配窗口的真实 forward）；
启动 profile/dummy 不占该窗口；开始门槛之前/请求未命中的 forward 也不占预算。
采集不会在失败时自动重新开启。长响应超过 128 步仍可能不完整，应据已知 forward 范围调整开始门槛；
调整后也不能把窗口外输出当已检查。CPU attention 逐组 heads 计算，单组输入估算上限 64 MiB；
每层快照张量载荷上限 64 MiB、每 worker 的 attention 快照累计默认 128 MiB。
文件容器开销、JSONL、既有 sampling 快照及临时内存不包含在该载荷预算内。
成功的大张量比较执行全量对照，但只落盘均匀选取的最多 2048 个元素；
FIA 偏差额外保留一个异常 head 的 key/value 及其 head 编号。
超 KV 长度、预算、保存容量都明确标记 skipped，不能算通过。

## 一轮结束后的验收

收集所有节点本轮完整目录，保留各 worker 原有目录名。
保存服务日志、完整 SHA、实际导入路径、算子包版本及 SSE 请求/响应，
将 SSE 请求与服务内 request ID 关联。只拉代码不会自动开启或完成验证。

例如本轮应采集 DP0–3 的 TP0，在汇总目录运行：

```bash
python tools/flashmla_check_diagnostics.py /absolute/path/to/collected-run \
  --expected-dp-ranks 0,1,2,3 --tp-rank 0 --require-history
```

- 退出 0：`SAMPLED_CHECKS_COMPLETE`，仅表示限定窗口/层/请求/位置的证据齐全且比较通过。
- 退出 1：`DISCREPANCY`，有数值、映射或采样异常，报告列出 forward、层和阶段。
- 退出 2：`INCOMPLETE`，包括缺 rank、只有 armed、缺 begin/end/张量文件、未命中历史、
  检查跳过或窗口耗尽。它不是“模型正常”。

针对零历史问题，还须确认报告中的 `zero_history_observed=true`，并在 `history_chunk` 记录中
确认实际出现了目标混合历史分段；没有命中时不能排除该假设。

脚本只解析 JSONL 并检查快照存在，不加载 pickle 或检验快照内容完整性。
`first_observed_anomaly` 表示同 worker 时间顺序中首个被观察的偏差，不等于根因。
模型输出/LM-head 快照和日志只含真实请求行，padding 不参与。
每个 worker 根目录保留 sampling 输出，`attention/` 下是新增 JSONL 与 CPU 张量。
数据可能包含请求身份和模型中间值，沿用本地诊断数据的保存方式。

一次复现若落在观测窗口和选中请求/层内，通常可判断问题首先在哪个被观察阶段出现。
若没有复现、异常层未被选中、上下文超预算，或者只有非选中 TP 出错，
仍需根据报告调整覆盖范围；不能承诺一轮就锁定根因。

## 本地回归

```bash
python -m pytest -q --confcutdir=tests/ut/worker/v2 \
  tests/ut/worker/v2/test_flashmla_chunk_diagnostics.py \
  tests/ut/worker/v2/test_flashmla_sample_diagnostics.py
```

CPU 用例包含非连续缓存、非零 offset、跨页尾段、零历史 NaN 输出与合法 `-inf` LSE、
混合请求、有限但错误的 writer/gather/FIA/merge、metadata 错位、预算与采集缺失。
新增回归包含目标窗口不被 smoke 耗尽、复用 metadata 清除、参考层之外的 NaN、
模型与 LM-head 行映射、padding 排除及新边界缺失。本次 worker 诊断 57 项、attention contract/lifecycle 44 项，合计 101 项 CPU 回归通过；
`bash format.sh ci` 全部通过。
真实 FIA/FlashMLA/NPU 行为尚未执行，必须由设备运行确认。

关于直接借鉴 1 号的理由、硬限制与仍不明确的地方，见
[逐项采用复核表](flashmla_plan1_adoption_review.md)。
