# 旧 B 线补充复核：差异、证据与待查项

本页整理本地旧 `flashmla_plan1_comparison.md` 第 86 行起的技术复核，供 PR #14 使用。
旧文档的 d7 部署状态、旧实验授权和通过记录不作为本候选的执行依据。
固定参考：1 号 `de31c53dc5b94ff246b17aa198404a082162c2f9`，旧 2 号
`a583897e0c67d9c23288686728124b04b511a462`，旧运行基点
`d7e950dc2a63436fc3c71bd8436533a18925da46`。
新候选完整 SHA 以 PR #14 的固定候选栏和本轮执行单为准。

| 复核点 | 实现区别与判断 | 本次补充 / 设备判据 |
| --- | --- | --- |
| writer 的位置 | 前处理先写共享 KV；FIA 当前块可直接消费新生成 K/V，后续 Decode 才读缓存。Prefill 输出正常不证明缓存正常 | Prefill/Decode 写前冻结 raw KV、norm weight、cos/sin，CPU FP32 独立计算 norm/RoPE，对照真实 slot 写后值；保留原 Prefill 返回 K/V 精确检查 |
| RoPE 分支 | 无 RoPE 时两边都有 scatter；有 RoPE 时 1 号普通分支可用独立 norm/RoPE/scatter，旧 B 保留 fused writer | 记录真实 use_mla_rope；没有走该分支不能归因 fused RoPE。参考用实际 cos/sin，尚未独立从 position 重算三角值 |
| 视图描述符 | 1 号也允许页轴非连续。单头分量的 singleton stride 可为 512/64，B 为 576；地址数学等价不证明二进制处理等价 | 新 scratch probe 同 backing 做 direct 与 squeeze/unsqueeze A/B，覆盖 fused/scatter、跨页 gather、短尾和未写区域；不是生产修复 |
| 非目标缓存污染 | 只核对目标 slot 会漏掉越界/误写 | 真实 writer 检查最多 32 个未写邻近 slot 及可用未写页哨兵，按字节检查以允许原有 NaN；不能覆盖整个缓存。scratch probe 另外检查完整 backing 的所有未写元素 |
| 零历史 | 1 号 active-row 过滤只在满足条件的 full-unabsorbed 路由；B FIA 分段保留空行 | 空 LSE=-inf 合法，空输出可未定义；要核对实际 merge 忽略其贡献。已有双 merge 参考与零历史恒等检查，必须实际命中混合历史 batch |
| 首 token | 首个可见 token 可能来自 Prefill FIA | 以 forward/request/phase 和 emitted 记录归属，不能按“第一个 token”直接判断 Flash Decode 出错 |
| padding | 两边都有零 used 行；两条 cu 在活跃前缀相同、padding 部分不同仍需二进制实测 | 保存实际 cu/used/lengths/page/slot/position；padding 不参与模型数值验收。实际 forward token 数不是客户端并发数 |
| 请求映射 | B 新增稳定 phase 排序，六项一起重排；CPU/device 镜像可同源一起错 | 在 gather 前独立从 request registry 冻结 req→state index、反向 ID、scheduled/computed/prefill，再对照排序后 batch；registry 自身也可能错误，不能当外部权威证明 |
| DP/MoE/dummy | chunk/mixed 改变 token 数可能改变原有通信选择；C60 或 running=15 不能证明每轮 15 token | 记录 execute_id、真实/padded token、DP counts、通信枚举、padding 摘要及有界 dummy/profile 入口。字段缺失即 unknown，枚举不证明实际 collective 内核 |
| eager buffer | 曾永久保留每种 eager 容量，ae92 修正已包含在 d7 | 不重复当待修问题；CPU 生命周期通过不排除 NPU 复用风险 |
| cos/sin 与 stream | B 在 metadata refresh 更新并等待，不是只生成一次；device_metadata.py 在三版相同，新变化是 builder 绑定与 submit/wait/release 接线 | wait 去重与 release 当前流构成条件风险，普通 eager 尚无已证实第二消费者；缺 record_stream 不能单独判错。用短 profiler 看真实生产/消费流及依赖 |
| sampler | 1/2 的 Gumbel 与 categorical 差异属于继承行为 | raw logits 已坏先查上游；同 seed 随机输出不同不能直接判数值错；当前原复现 temperature=0 |

主线已见的任务优先级/排序变动，尚不能认定修复了本例。当前没有因上述怀疑改动
生产 chunk 划分、空行筛选、merge、writer 算法或执行流策略；新增的是可关闭的诊断。
KDA state 的独立参考、完整投影/MLP/MoE/LM-head 重算、未选 TP/query/层、
MRv1、图 replay、speculative rejection 均未被本轮 eager 采样验证覆盖。
同步诊断会改变时序；开启后不复现不等于修复。

## 设备独立描述符实验

先保留原服务复现证据；若 writer/gather 或布局指向描述符，再在现场 owner 确认的空闲卡执行。
该脚本只分配合成 BF16 scratch，不读取生产 KV；无 `--execute` 不执行设备算子。

```bash
python -m tools.flashmla_writer_probe --execute --device npu:0 \
  --page-stride 81408 --output /absolute/path/to/run/writer-probe.json
```

默认页 stride 81408、token stride 576、storage offset 32；不要把人工 token stride=1152
的失败直接解释为生产 stride=576 的失败。报告四种 writer/视图组合、目标写入参考、
全部未写 backing、gather 的 start/length（0/130、128/2、128/1）与只读性。
未执行、异常中止或缺某组合都不能声称 A/B 通过；全部通过也不证明生产 allocator/并发正确。

关联：[采用理由表](flashmla_plan1_adoption_review.md)、
[采集手册](flashmla_chunk_diagnostics.md)、[四机与 profiler](flashmla_four_node_validation.md)。
