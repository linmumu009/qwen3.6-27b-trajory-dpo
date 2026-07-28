# qwen3.6-27b-trajory-dpo

Qwen3.6-27B 轨迹偏好训练工程。目标不是只证明 DPO 能运行，而是在冻结数据、对照实验和独立评测下，稳定提升 PI agent 的任务完成率、结果正确性与工具调用质量。

## 记录规则

- 每次实验、数据变更、训练配置变更、评测或结论更新，都必须同步更新本 README 的“实验日志”。
- 每个正式实验在启动前冻结：数据哈希、起始 checkpoint、训练参数、随机种子、评测题集和成功门槛。
- 不以 training loss 或 DPO margin 单独证明模型质量提升；必须与 matched continued-SFT、普通 DPO 和独立 heldout 对照。
- 不把缓存/推理性能收益混同为模型权重质量收益。
- 服务器登录方式、私钥说明和含内部证据的原始报告不提交到 GitHub。

## 当前结论

截至 2026-07-28，工程和证据已经足够进入受控 pilot，但不适合直接开展无对照的全量普通 DPO。

已有 21 条件、3 个训练随机种子、每条件 192 次 rollout 的冻结确认实验表明：

| 条件 | Full success | Exact result | Process score |
|---|---:|---:|---:|
| chosen-only SFT | 76.91% | 76.91% | 7.1215 |
| continued SFT | 76.22% | 76.56% | 7.0017 |
| warm-start DPO（75 steps） | 67.01% | 80.03% | 6.8924 |
| verified RPO（DPO + chosen NLL） | **81.77%** | **82.12%** | **7.4080** |
| randomized RPO | 65.62% | 66.67% | 6.0278 |

verified RPO 相对 matched continued-SFT：

- full success：`+5.56` 个百分点，95% CI `[+1.04, +10.42]`
- exact result：`+5.56` 个百分点，95% CI `[+1.04, +10.59]`
- 三个 seed 的 full-success 增益：`+6.25 / +4.69 / +5.73` 个百分点

因此下一轮以 **chosen-SFT warm start + verified RPO** 为主线；普通 DPO保留为对照，不作为默认训练目标。

## 已核实资产

### 5 号机

- 原始老板 PI 容器：`rjx-vllm-qwen36`，当前为退出状态。
- Qwen3.6-27B v15-table 轨迹：1500 条，DWH / KB / Hybrid 各 500 条。
- 对应 verdict：1500 个唯一任务，其中：
  - correct：468
  - partial：391
  - incorrect：125
  - incomplete：516
- 同任务还有 Qwen3.7-Max、DeepSeek-v4-Pro、GLM-5.2 轨迹和 verdict。
- 四模型在 1500 个共同任务上可形成 871 个严格序偏好任务；其中 669 个满足 `partial/correct > incorrect/incomplete` 的强分离。
- 现有 DPO/RPO 工程位于服务器项目 `llin-rl-dpo-p2`。
- 16 张 NPU 当前 AICore 均为 0%，但 DPO 容器仍运行 NPU 显存保留进程；正式实验前需受控释放。

### 6 号机

- 已生成并校验 450 对 `correct > partial/incorrect` 的严格偏好对，覆盖 v15/v20/v21。
- 已生成 16K、32K、36K、40K、48K、100K 长度桶及 manifest/hash。
- 40K 稳定候选包含 141 个唯一 prompt：
  - v15：50
  - v20：51
  - v21：40
  - KB：62，Hybrid：56，DWH：23
  - `correct > partial`：93
  - `correct > incorrect`：48
- 40K 已使用 `TP8 × PP1 × CP2 × SP` 完成 worst20、checkpoint 严格恢复验证。
- 48K 仅完成固定 worst1 的 500-step 压力测试；100K 属于无安全余量的极限配置，不用于正式 pilot。

## 数据质量注意事项

- qwen3.6 v15-table verdict 文件为 1500 个唯一任务，可用于严格序偏好构造。
- 对应 reward 文件存在一个重复键 `task_000147`，并缺少 `task_000033`。当前 450 对数据按严格 verdict 构造，不受该问题阻塞；任何 reward 加权实验前必须先修复并重新哈希。
- timeout/incomplete 不能自动当作语义负样本；需要排除基础设施超时、服务崩溃和工具环境错误。
- 多模型偏好对可能携带模型风格、长度和工具调用次数偏差，训练前必须输出来源模型与长度方向审计。

## 下一轮实验：P-001 轨迹 RPO pilot

状态：`preparing`

### 目标

验证老板 PI 轨迹上的 verified RPO 是否能稳定超过：

1. 同起点 chosen-only/continued SFT；
2. 普通 warm-start DPO；
3. randomized-label RPO。

### 冻结方案

- 主训练上限：40K。
- 工程拓扑：`TP8 × PP1 × CP2 × SP`，16 张 Ascend NPU。
- 初始候选：Qwen3.6-27B base 与现有通用 reasoning-SFT；先用冻结 PI 基线评测选择，禁止仅凭 checkpoint 名称指定。
- 每个偏好条件的 optimized policy 与 reference policy 使用同一个 matched chosen-SFT 起点。
- 主目标候选：
  - DPO beta：`0.1`
  - RPO alpha：`1.0`
  - learning rate：首轮从 `5e-5` 开始
  - LoRA：`r=8, alpha=32`
  - BF16、full recompute
- 先做 1-step 和短程 smoke；通过数值、显存、checkpoint 和恢复门槛后才启动正式训练。

### 评测门槛

- 冻结全新 PI heldout；不使用已经观察过的 21 道缓存实验题做模型选择。
- 质量指标：full success、exact result、工具调用成功率、答案 grounding、process score。
- 单独报告 DWH / KB / Hybrid。
- 三个训练 seed，报告逐 seed 指标和 task-level paired bootstrap 区间。
- 推理环境固定使用稳定版本，所有条件使用相同缓存、MTP、采样参数和超时上限。
- verified RPO 必须同时超过 continued-SFT 与普通 DPO，且不能依赖单个 seed。
- 保留通用能力回归集，防止 PI 指标提升但通用推理能力退化。

## 实验日志

### 2026-07-28：E-000 资产与证据审计

更新内容：

- 核实 5/6 号机登录、容器、NPU、磁盘和活跃进程状态。
- 定位老板 Qwen3.6-27B v15-table 的 1500 条轨迹、verdict、reward 以及三个外部模型的同任务轨迹。
- 审计四模型任务键和 verdict，确认 1500 个任务完全对齐，871 个任务存在严格序偏好。
- 定位 6 号机已处理的 450 对严格偏好数据及各长度桶。
- 核实 40K worst20、严格 checkpoint 恢复、48K/100K 压力测试边界。
- 复核 12 个普通 DPO run 的 likelihood displacement：chosen/rejected likelihood 同时下降，仅 margin 增大。
- 复核 21 条件确认实验：verified RPO 相对 continued-SFT 获得稳定、区间为正的 heldout 提升。
- 确认下一轮不直接复用普通 DPO，而以 matched chosen-SFT warm start + verified RPO 为主线。

结论：

- 数据、工程和目标函数证据足够进入 P-001 pilot。
- 正式开训前仍需冻结独立 heldout、起始 checkpoint 和数据拆分，并释放 5 号机的显存保留进程。
