# KDA / MLA 联合修复：逐步验证

本次接在 #15 `c9437d55` 配置透传修复之后，保留 `8340256f` 诊断实现。配套 vLLM 保持
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`，Plan1 对照为 `de31c53d`。
PR #13 `93dbf8f9` 已在第七轮出现异常，保留为失败对照。新候选的完整 SHA
以交付记录为准，不能只记录分支。以下设备步骤尚未执行。

## 第一步：确认物理布局

保留已修复的 descriptor/page-major 协议：同页 conv/recurrent、真实 manager
pitch、页内 offset、原 storage offset、payload 边界与 MLA 子页比例。
已有 CPU/meta 用例重建旧报告槽3/页174的精确地址；本次另补当前几何下的
槽3/页174真实算子保护用例。它不等于重新执行旧 d7 的849页事故。

独立 Mamba 也保持当前上游 page-major 合同，没有照搬旧报告的独立池连续分支。
不把已修正的 descriptor 重新降成旧 state-major。运行前保存每层真实 spec、
planner descriptor、tensor shape/stride/data_ptr/storage_offset 和有效块映射。
共享 storage 不是自动冲突；按当前有效所有权及字节范围判断。

## 第二步：逐个验证 KDA 状态读写

现用 FLA conv、native recurrent、Triton selected-state gather/scatter。
与 Plan1 的自带 conv/native state-copy/A5 融合数值路径存在实现差异，本次
没有盲目替换这些算子。锁定实际 CANN/FLA/OPP/FlashMLA 文件 hash 后执行：

```bash
python -m pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_hybrid_page_state_npu.py
```

本次新增或加强：

- 冷 slot1 预填 NaN，初始标志为 false；warm slot3 保留，测试真实 conv、chunk、
  recurrent 的输出、选中状态与 dense 对照、其他活跃 MLA 页的字节保护。
- COW 用 uint8 字节比较，包含 NaN 位型、无穷与负零的 CPU 复制回归；计算结果
  仍检查数值和有限性。不采用全局 equal_nan，也不屏蔽真实非有限计算结果。
- 同批历史 `[0,129]`、`[1,257]`、`[257,1]`，query 长度1/2，manager128/384/768；
  writer→KDA复用脏页→FIA分块→KDA→两步 FlashMLA，与独立 FP32 attention 对照。
- 当前几何下，真实 slot3 更新不得破坏仍有效的 MLA 页174。

NPU fixture 仍是受控输入和 per-layer descriptor，不代表三种 planner 布局、
实际 scheduler 或全模型验收。失败时保存原始输入、目标及相邻字节和首坏算子；
不能只改容差或跳过断言。

## 第三步：对齐 Plan1 的混合历史语义

FlashMLA 开启、普通 CP1 的 chunked prefill 在构建 metadata 时计算有历史的
query 行。当前段先建立各行输出/LSE；每个历史块只对仍有 KV 的行调用 FIA
并合并，历史为空的行保持原结果，整块为空则跳过。KV gather 的排列不变，
删掉零长度段无需重新搬运 KV。索引每批建立一次，不在每层做设备到主机同步。
PCP/DCP 与未开启 FlashMLA 的路径保留原处理，仍需各自验收。

CPU 用 production orchestration、独立完整 attention 参考以及“空 KV 返回 NaN”
的对抗算子验证旧路径失败、新路径通过。它证明此类污染可被隔离，不证明真实
FIA 一定返回 NaN，也不证明这就是 #13 第七轮根因。真实 NPU 用例是必经步骤。

## 第四步：同一服务第七轮取证

新配置 `additional_config.bline_diagnostics.arm_file` 可指定绝对路径。
路径应位于本次运行独有目录，启动前不存在，所有 TP worker 看见相同文件系统。
没有此键则保持原来首个真实请求开始采集。工具 `plan` 也接受 `--arm-file`。

将以下对象合入原有 additional-config，保留其他配置；替换路径为本次真实目录：

```json
{
  "bline_diagnostics": {
    "enabled": true,
    "arm_file": "/absolute/path/to/unique-run/round7.arm",
    "max_steps": 32,
    "max_bytes": 1073741824,
    "max_events": 16384,
    "max_pages": 4096,
    "max_requests": 64
  }
}
```

1. 保留 #13 最新失败的 dummy5、20K配置、TP8/DP1、eager、语料及完整启动参数；
   不用4层或4K替代同条件定位。记录软件/二进制/模型/语料 hash。
2. 同一服务执行原前六轮 C60×8。标记文件不存在时不扫描张量、不消耗 step/byte
   预算；开启诊断安装的 wrappers 和文件检查仍可能轻微改变时序。
3. 等第六轮所有请求结束，在第七轮发出前创建标记：

   ```bash
   touch /absolute/path/to/unique-run/round7.arm
   ```

4. 第七轮每个 worker 的首个真实执行步骤看到文件后记录 `armed`，再开始有界
   采集。warmup/idle 不触发。不得重启服务、释放缓存或重放成单独请求。
5. 保存全部8个 TP rank 的 armed/begin/映射/检查/step_summary，以及完整客户端
   输入、实际 token 长度、请求 ID、顺序和输出。删除标记不会关闭已启用采集，
   也不会重置预算。用新运行目录避免旧标记导致过早启用。

预算仍可能不足覆盖整轮；1GiB 不是覆盖保证。达到限制必须记 UNCOVERED，
不是后续请求检查通过。未触发历史 prefill 的轮次不能算该路径通过，可单独
执行第二步的受控用例。关闭诊断后重跑同负载，以评估同步/采集时序影响。

```bash
python tools/bline/validate.py summary \
  --expected-ranks 0,1,2,3,4,5,6,7 \
  --output /absolute/path/to/unique-run/summary.json \
  /absolute/path/to/unique-run/service.log
```

先坏在 KDA 前后历史字节→查地址/算子写入/复用；writer 回读先坏→查写入映射；
历史保护通过但 attention 先坏→抓真实 Q/KV/table/length 与 FIA/FlashMLA；
attention 有限但 hidden/logits 先坏→查后续模型与量化。现有探针不直接检查
masked LSE、所有 writer 的非目标页，不能把未覆盖区域排除为根因。

## 第五步：原报告负载与验收

本次诊断仍限 DP1/PP1/CP1 eager，无 speculative/KV transfer/batch sharder；
DP4 会明确报告未覆盖。完成局部定位后恢复完整真实 W4A8/MXFP/Quarot 权重、
四节点每节点 TP8/总 DP4，关闭此 DP1 探针，跑 C60×8 至少10轮、长短混合、
块回收/COW/zero、低并发及显存/时延对照。要在 DP4 抓同类首坏证据，仍需
另行支持该拓扑，不能把单机探针日志冒充四节点证据。

PR #13/#15 都含页修复，两者 A/B/A 只比较集成背景。原始 d7 对照需单独锁定其
配套环境并说明变量差异。HTTP200、有限值或无感叹号均不能单独代替真实权重
数值对照和完整验收。独立 KDA/MLA 缓冲区依然是尚未实施的备选，使用前重算容量。

## 本地验证边界

CPU worker 回归119项、FlashMLA合同/metadata回归44项通过；包含地址/descriptor、
复制位型、冷状态、混合历史数值及延迟启用测试。本地无 vLLM/torch_npu 和 NPU，
未运行完整 UT/ST、真实 FIA/FLA/native 算子、实际 ModelConfig、服务或性能验证。
