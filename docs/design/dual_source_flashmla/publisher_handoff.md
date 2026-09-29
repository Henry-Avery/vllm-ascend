# PR 15 发布机验证交接

本轮优先验收 #15；#13 `93dbf8f91c0ca3d94e5617b0a1903b0456edc40a`
保留为修复对照。VA 使用本次交付回报中的完整 SHA，配套 vLLM 固定为
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。不重合移动的上游 head，
不只记录分支名。本文是待执行步骤；本地未运行 NPU、构建或服务。

## 1. 锁定实际运行来源

在发布机原有 checkout 正常拉取 `codex/dual-source-flashmla`，核对交付 SHA、
vLLM SHA、工作区改动与实际 Python import 路径。记录 torch/torch_npu、CANN、
FLA wheel/native OPP、外部 FlashMLA wheel/OPP 的版本与文件 hash。
保留原机器启动命令、模型配置及硬件拓扑；不要同时更换依赖以免失去对照。

## 2. 先复现同一启动条件

沿用原事故 dummy 5 层、eager、TP8/DP1、无 speculative 的完整命令；
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

## 5. 可执行 B 线减层诊断（本次新增）

本节使用本次交付的新完整 VA SHA（PR 正文和交付 manifest），父提交为
`029e21ea5e68cc218bd26e74a7550bf5657e6a84`；vLLM 仍固定 `ced6857a`。
这是诊断实现，尚无真实设备通过结论。

首版支持真实 V2 eager、TP8/DP1、PP1/CP1、无 speculative、无 KV transfer、
无 batch sharder。`additional_config.bline_diagnostics.enabled` 默认为 false；
关闭时不安装执行/算子/sample hooks，不 clone、不同步、不扫描 tensor。
首版 DP4、图模式、其他 cache 表示会显式未覆盖；最终 DP4 压测关闭此探针。

开启后只对有效请求 metadata 中 `seq_len-query_len` 的历史 token payload
做 KDA 调用前后精确字节比较。忽略空闲页、当前待写 token、PAD slot；conv
的 null0 不当写入，chunk 的 keep_meta 按实际选择过滤。原始 descriptor、
最终 tensor stride/offset、group/request/state/manager/kernel 映射逐条记录。
同一共享 storage 并不自动意味着冲突；冲突判断使用有效页表、manager 比例和
当前 state ID。冷 prefill 无历史记 NOT_APPLICABLE，不能提供保护通过证据。

K3 `_exec_kv_no_rope` writer 用真实归一化后返回的 latent/rope 对有效 slots
逐元素回读。其他融合/量化 writer 未触发此入口时，汇总必须保留未覆盖。
FIA 和 external FlashMLA 当前检查公开 eager 方法返回的 attention 输出有限性；
不检查 masked LSE，不声称与独立数值参考一致。普通/lmhead-TP sample 均在原始
compute_logits 返回后、grammar/sampler 前检查有效 logits 行及其 final hidden，
padding 行排除。首坏事件带 rank/layer/step/映射或首坏元素/字节位置。
writer 对非目标页的额外保护、算子独立数值对照和整个模型的准确率仍是独立门槛。

### 减层计划与启动模板

在发布机实际安装环境、VA checkout 中执行。MODEL 指向原模型，保持原权重目录。
工具通过实际 `ModelConfig` 先读取配置，再用嵌套 hf-overrides 生成减层有效配置；
比较全部 text_config 字段，除了层数不得变化。验证 `is_kda_layer(index)` 的真实
语义（配置 kda_layers 为 1-based，日志 layer index 为 0-based）。若原前缀不是
KDA0/1/2 + MLA3，工具直接拒绝，不盲切。

```bash
MODEL=/absolute/path/to/original-model
OUT=/absolute/path/to/bline-run
mkdir -p "$OUT"
python tools/bline/validate.py plan \
  --model "$MODEL" --layers 4 --steps 8 --max-bytes 268435456 \
  --output "$OUT/plan.json" -- \
  --max-num-seqs 64 --max-model-len YOUR_ORIGINAL_MAX_MODEL_LEN
```

将最后一行的占位值替换为原值，并在 `--` 后追加原有量化、缓存、服务端口等参数；
工具不改这些参数，也不会猜测它们。已有 additional-config 请用
`--additional-config /path/to/existing.json`（放在 `--` 前）合并；不要重复传同名 CLI。
原 5 层 smoke 可改 `--layers 5`，但仍须先通过相同层类型检查。
若模型需要 remote code，显式传 `--trust-remote-code`。工具只生成计划、不启动服务。

从 plan.json 提取 command 到 launch.sh，审核后在发布机运行：

```bash
python - "$OUT/plan.json" "$OUT/launch.sh" <<'PY'
import json
import sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(plan["command"] + "\n")
print(plan["effective_layers"])
PY
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ASCEND_ENABLE_FLASH_MLA=1 \
  bash "$OUT/launch.sh" > "$OUT/service.log" 2>&1
```

dummy 路径 smoke 可在 plan 的 `--` 后加 `--load-format dummy`；它不代表真实权重
结论。服务内部 warmup/profile/idle 步骤不消耗诊断步数；真实 dummy-weight 请求仍
消耗预算，因为它实际执行了模型路径。然后用真实权重减层，保留原维度、量化、TP
和缓存参数，并从日志确认至少一对 KDA/MLA 共享 storage 且 manager 映射合法。
每个请求 decode 至少 2 步，使实际历史被读取；还要有 history-backed 分段 prefill，
不能只有冷 prefill+decode 就声称 chunk 保护已经命中。

### 预算、汇总与判据

预算均为单 worker、跨本次 run 的累计限制，配置键及范围：

| 键 | 默认 | 合法范围 |
| --- | --- | --- |
| max_steps | 8 | 1–128 个真实请求步骤 |
| max_bytes | 268435456 | 1–1073741824 个采集 payload 字节 |
| max_events | 2048 | 1–16384 条事件 |
| max_pages | 256 | 1–4096 个单次保护页 |
| max_requests | 64 | 1–256 个请求 |

这些预算用于限制采集量，不是峰值内存或耗时的保证。C60 全词表 logits 一步可能
约 40MB，多层历史的前后快照更大；默认预算很快耗尽是预期行为，不能冒称检查了
后续十轮。先用独立 run 采集 C1，再重启独立 run 采集 C4，再为 C60 选择短采集窗口
和明确预算。不要在 C1/C4 耗尽预算后直接把 C60 当已检查；每轮日志单独归档。
同步/拷贝会改变时序，关闭诊断必须重跑同一负载。

```bash
python tools/bline/validate.py summary \
  --expected-ranks 0,1,2,3,4,5,6,7 \
  --output "$OUT/summary.json" "$OUT/service.log"
```

日志可以分 rank 保存后一起传入。每个预期 rank 必须恰有一个 run_id、非空且完整的
binding/begin/step_summary、真实 protected_mapping，以及每层跨步骤的必要 CHECK。
合法未发生的 phase 和 cold history 不算通过证据；需要后续实际命中。
真正 budget/unsupported/缺rank/缺路由/异常截断不能被吞掉。

- FAIL（退出1）：有效 payload 改变、有效 writer 回读不等、非法索引/所有权冲突，
  或有效 attention/hidden/raw logits 非有限。保存首坏事件和对应服务日志。
- UNCOVERED（退出2）：预算、支持范围、输入或日志不足；必须补跑，不能当通过。
- CHECK（退出0）：仅表示所有预期 rank 的有限预算诊断证据齐全，无该探针检测到的
  异常；仍不是独立数值对照、真实权重语义正确或整网验收 PASS。TRACE 单独没有数值效力。

### 固定来源与二进制 manifest

```bash
python tools/bline/validate.py manifest \
  --va-repo /absolute/path/to/vllm-ascend \
  --vllm-repo /absolute/path/to/vllm \
  --cann YOUR_ACTUAL_CANN_VERSION \
  --binary /absolute/path/to/loaded/fla-native-binary \
  --binary /absolute/path/to/loaded/flashmla-opp-binary \
  --output "$OUT/manifest.json"
```

追加所有实际加载的相关库/OPP 文件，工具记录 SHA256；包发现失败保留 null，
需从发布机实际加载路径补齐，不能推断已加载。记录所有 rank 日志、启动命令、
原/有效配置 hash、输入与输出、硬件拓扑；不得只记录 branch 或 wheel 名。

减层改变调度与可用缓存容量。诊断关闭后重跑减层 C1→C4→C60，再恢复原整网配置、
TP8/DP4、C60×8 至少10轮长短混合/分段prefill/复用压力。恢复时移除减层 hf-overrides
和诊断配置，不要把四层结果当整网验收。真实 NPU seam、独立数值参考、图模式和性能
各自记录；HTTP200、无感叹号或仅有限性仍不足以放行。
