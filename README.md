# qwen3.6-27b-trajory-dpo

Qwen3.6-27B 轨迹偏好训练工程。目标不是只证明 DPO 能运行，而是在冻结数据、对照实验和独立评测下，稳定提升 PI agent 的任务完成率、结果正确性与工具调用质量。

## 记录规则

- 每次实验、数据变更、训练配置变更、评测或结论更新，都必须同步更新本 README 的“实验日志”。
- 每个正式实验在启动前冻结：数据哈希、起始 checkpoint、训练参数、随机种子、评测题集和成功门槛。
- 不以 training loss 或 DPO margin 单独证明模型质量提升；必须与 matched continued-SFT、普通 DPO 和独立 heldout 对照。
- 不把缓存/推理性能收益混同为模型权重质量收益。
- 服务器登录方式、私钥说明和含内部证据的原始报告不提交到 GitHub。
- Windows 只作为 SSH 控制端，不承载模型、checkpoint 或评测数据中转；5/6 号机的数据面必须走服务器内网。
- 只允许使用或新建 `llin-*` 容器和镜像；不得使用、修改或覆盖其他人的容器与镜像。

## 当前结论

截至 2026-07-29，工程和证据已经足够进入受控 pilot，但不适合直接开展无对照的全量普通 DPO 或长跑在线 GRPO。

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

多机工程方面，`5 号训练 + 6 号在线 rollout` 的同步 GRPO 已从半机 8 卡扩展到两台各 16 张计算 NPU，并完成 40K 上下文的真实多提示单步闭环：5 号采用 `TP8 × PP1 × CP2 × SP`，6 号采用 `TP8 × DP2`；初始 LoRA 由 5 号直接 rsync 到 6 号，经 SHA256、版本目录和原子 symlink 发布后由全部 rollout worker 确认加载。最新修正版单步使用 prompt 2/10，各生成 8 条轨迹，reward 为 `0×4 / 0.2×3 / 1×9`，16/16 advantage 非零且 finite；关闭 1-step warmup 后，更新 checkpoint 的 LoRA-B、Megatron BF16 model 和 FP32 optimizer master 均确认非零且 finite，HF adapter SHA256 变为 `eddb8468ac5874eeef7931adf5bb803573867bd1ecab04faedabb5449ff64cf8`。该结果证明 40K 全机工程闭环、有效优化信号和真实参数更新，不等同于独立 heldout 上的模型效果提升。此前把 warmup=3% 的 1-step run 记为“权重已更新”是不准确的：它完成了反向和 checkpoint，但首步实际学习率为 0，LoRA-B 未变化。

对这 136 条历史轨迹进行 v2 反事实重放后，确认纯终局 reward 过稀疏：只有 4/17 组具有方差；保守混合 reward 为 12/17 组保留方差。当前在线 GRPO 默认契约因此改为：只训练 assistant 内容与 tool call，工具执行结果只作为下一步观察而不进入 loss；`1.0` 只奖励安全、协议有效、成功查询必需表且命中 gold evidence 的终局答案，`0.2` 只奖励同样满足前置条件但尚未命中 gold 的终局验证进展；截断、仅安全调用、仅有答案或一般工具成功均为 `0`。不采用“每一步都给分”的稠密过程奖励，除非后续获得独立、校准过的 step verifier。

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
- 16 张计算 NPU 当前没有 NPU 进程；自有全机 trainer `llin-qwen36-grpo-trainer-m05-p001-maxctx-0729` 保持运行但仅有 `sleep infinity`，不占用 NPU；已完成的旧 DPO 容器保持停止。

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
- 自有全机 rollout 容器 `llin-qwen36-grpo-pi-rollout-m06-maxctx-0729` 可见 16 张计算 NPU；当前仅有 `sleep infinity`，28220/28221 无监听且没有 NPU 进程。

## 数据质量注意事项

- qwen3.6 v15-table verdict 文件为 1500 个唯一任务，可用于严格序偏好构造。
- 对应 reward 文件存在一个重复键 `task_000147`，并缺少 `task_000033`。当前 450 对数据按严格 verdict 构造，不受该问题阻塞；任何 reward 加权实验前必须先修复并重新哈希。
- timeout/incomplete 不能自动当作语义负样本；需要排除基础设施超时、服务崩溃和工具环境错误。
- 多模型偏好对可能携带模型风格、长度和工具调用次数偏差，训练前必须输出来源模型与长度方向审计。

## 下一轮实验：P-001 轨迹 RPO pilot

状态：`data-frozen / multihost-data-plane-passed / trajectory-grpo-v2-engineered / maxctx40k-multiprompt-real-update-1step-passed / no-long-run-authorized`

### 目标

验证老板 PI 轨迹上的 verified RPO 是否能稳定超过：

1. 同起点 chosen-only/continued SFT；
2. 普通 warm-start DPO；
3. randomized-label RPO。

### 冻结方案

- 主训练上限：40K。
- 工程拓扑：`TP8 × PP1 × CP2 × SP`，16 张 Ascend NPU。
- 起始策略：Qwen3.6-27B base 先在 101 条 chosen 轨迹上做 matched PI-SFT；不直接混入现有通用 reasoning-SFT，避免把不同训练历史混入因果对照。
- 每个偏好条件的 optimized policy 与 reference policy 使用同一个 matched chosen-SFT 起点。
- 主目标候选：
  - DPO beta：`0.1`
  - RPO alpha：`1.0`
  - learning rate：首轮从 `5e-5` 开始
  - LoRA：`r=8, alpha=32`
  - LoRA target modules：`linear_qkv + linear_fc1`，沿用老板动态 rollout 已验证契约；不再使用 `all-linear`
  - BF16、full recompute
- 正式单 seed pilot 计划在同一 PI-SFT 起点比较：
  - continued-SFT：75 steps
  - verified DPO：保存并检查 25 / 50 / 75 steps
  - verified RPO：75 steps
  - randomized-label RPO：75 steps
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

### 2026-07-28：E-001 P-001 数据冻结与运行环境复刻

更新内容：

- 新增 `scripts/freeze_p001_split.py`，逐行验证 preference 与 manifest 对齐、chosen/rejected prompt 前缀一致、任务键唯一和 train/heldout 无 prompt 泄漏。
- 固定 v15+v20 为训练集，共 101 对；固定 v21 为内部 heldout，共 40 个 prompt；两者 prompt SHA256 重叠为 0。
- 训练集构成：
  - 版本：v15 50、v20 51
  - 类型：DWH 20、Hybrid 37、KB 44
  - 偏好：`correct > partial` 68、`correct > incorrect` 33
  - 长度方向：chosen 更长 60、rejected 更长 41
- 新增 seed-42 的平衡随机标签对照：101 对中固定交换 50 对 chosen/rejected，保留 prompt 与样本边际分布。
- 新增训练集中真实最坏长度的 preference/SFT worst1 烟测输入。
- 新增 `docker/Dockerfile.megatron-dpo-runtime`，精确复刻 6 号机已通过 40K 测试的 Python 运行层。
- 新增 `scripts/run_p001_megatron_inner.sh`：
  - 支持 base PI-SFT、continued-SFT、DPO、RPO、randomized-RPO；
  - 固定 `TP8 × PP1 × CP2 × SP`、40K、LoRA r8/alpha32；
  - 对每次 run 保存环境版本、输入哈希、NPU 前后状态和退出码。
- 新增 `scripts/create_p001_container.sh`，默认使用非 privileged、host network/IPC，仅显式映射 16 张 NPU 与必要 Ascend 管理设备；不挂载整块 `/data3`。
- 仓库准备阶段已将 6 号机约 65MB 的压缩运行补丁包和冻结数据放入 Git 忽略目录；未搬运 18.7GB 整镜像，该动作不属于 5→6 模型交接数据面。
- 首次容器 import 自检定位到两个隐含条件：容器工作目录必须避开基座自带 Megatron；且 `transformer-engine-2.14.1` 目录只是空 PyPI 元包，不能加入 `PYTHONPATH`。Ascend 路径由 NPU 可见时的 MindSpeed adaptor 注入；元包归档只保留作来源审计。

冻结哈希：

| 产物 | SHA256 |
|---|---|
| 40K 源 preference | `7e108af5e6f5695804159c997ca54350febd8e5f3eac2055a613466518a88d17` |
| 40K 源 manifest | `dfde4e51037a0dabf14abbe5c37edbff9c5052c396d9defb8ed2add9da95eeac` |
| P-001 train preference（101） | `6057a2cb70fb28cf23ba4b2477bde703eb4eb62beb639ee4a54c42b4c6ee6dd4` |
| P-001 chosen SFT（101） | `eba20b9aacdf0de22fc09f5326306543d8c80b318c66a2e5ba9525270bb5f80c` |
| P-001 randomized preference（101） | `da62b39d532ee13ebf6b32683965e71f6e8d8619d97d3b05173e0e34af038695` |
| P-001 internal heldout tasks（40） | `ecd6dfd35ee8d29af475bcc33566186bc0b4c888e0e5c585f300de4c5ac4c99e` |
| 运行补丁归档 | `fe77cad7a3068257e5f51f1ec5901b29d41dc4bd86f3f5eeba05df0ad15aff10` |
| transformer-engine 补丁归档 | `62fdd248d89da3cff3b1ef50a1b6a4d892634424d1da52a7992728b39067070f` |

结论：

- P-001 的训练/内部 heldout 已冻结，可进入 5 号机 40K worst1 烟测。
- v21 属于严格未参与训练的内部 heldout，但团队已经接触过该版本，不能代替最终的全新 PI 任务集；它适合 pilot 选型，不足以单独支撑对外泛化结论。

### 2026-07-28：E-002 5 号机最小权限容器自检

更新内容：

- 在 5 号机本地 `mindspeed-llm:26.0.0-a3-sshd-yehairui` 基座上成功构建 P-001 镜像：
  - image ID：`sha256:20f3fc105341b6e13ed3a232aed1a87eac2b58cca44d47dc5174e99526dbcbc4`
  - Python：3.11
  - Transformers：5.12.1
  - Swift：4.5.0.dev0
- 为避免扩大宿主权限，依次测试三种非 privileged 配置：
  1. bridge network + private IPC + 19 个显式 Ascend 设备；
  2. bridge network + host IPC + 19 个显式 Ascend 设备；
  3. host network + host IPC + 19 个显式 Ascend 设备，并设置完整 physical/visible device IDs。
- 三种配置均返回：
  - `torch.npu.device_count() == 0`
  - `torch.npu.is_available() == False`
  - DCMI：`device is used, ret=-8020`
- 对照检查显示，同机 `slime-qwen35-rl-dev` 在非 privileged 模式可见 16 卡；而本机 MindSpeed 26 系列现有可见 NPU 的容器均为 privileged。故障边界位于 MindSpeed 26 基座与本机 Ascend runtime/driver 的权限组合，不是 P-001 数据或 trainer 参数。
- 已确认旧 DPO 训练有 `TRAINING_COMPLETE` 标记，并受控停止：
  - `llin-rl-dpo-p2-formal-0-3`
  - `llin-rl-dpo-p2-formal-4-15`
  它们没有被删除，可随时恢复。
- 停止后宿主 `npu-smi` 显示 16 张卡 AICore 均为 0，且没有 NPU 进程。
- 自检失败的 `llin-qwen36-p001-megatron` 也已停止；镜像、容器定义、冻结数据和运行目录均保留。

结论：

- 5 号机资源确实空闲，但当前安全的非 privileged 容器无法使用这套 MindSpeed 26 运行层。
- 下一步需要二选一的明确授权：
  1. 在 5 号机创建 privileged P-001 容器，但仍只挂载 P-001 工作目录、只读模型与必要 Ascend 路径，不挂 Docker socket、不挂整块数据盘；
  2. 改在 6 号机已经验证的 privileged 长上下文容器中执行 smoke/pilot。
- 在获得选择前不启动训练，也不把“能 import/能起容器”误记为 smoke 通过。

### 2026-07-28：E-003 训练/推理解耦方案

决策：

- 5 号机专用于 P-001 训练。
- 6 号机专用于 checkpoint 推理、PI agent rollout 和 heldout 评测。

核实结果：

- 两台机器的 16 张 NPU 均为 AICore 0%，无 NPU 进程。
- 6 号机已有多个 Qwen3.6 rollout/GRPO 容器，但当前都只运行 `sleep infinity`。
- 6 号机已有 Qwen3.6-27B base（约 52GB）和 PI sandbox 挂载。
- 已验证的 rollout 镜像：
  - image：`llin-vllm-ascend:grpo-pi-deps-20260727`
  - vLLM：0.23.0
  - Torch：2.10.0
  - 单容器可见 8 张 NPU
  - 支持 `--enable-lora`、`--lora-modules`、LoRA rank 8 和 tensor parallel
- 6 号机 8000-8015 当前无监听服务，可冻结独立评测端口。

交接协议：

1. 5 号机每个训练条件只输出 LoRA adapter、`args.json`、输入哈希和训练日志。
2. 每个 checkpoint 计算 SHA256，由 5 号机通过内网 `192.168.202.5 → 192.168.202.4` 直接 `rsync/SSH` 到 6 号 `.incoming`；Windows 不进入数据链路。
3. 6 号逐包和逐文件校验，通过后在同一文件系统内原子发布到 `adapters/`，禁止覆盖同名产物。
4. 6 号机用同一个 base、同一 `llin-*` 推理镜像和完全一致的采样/缓存/超时参数评测不同条件。
5. 离线 DPO/RPO checkpoint 默认先合并 LoRA 再评测；静态 PEFT LoRA 只有通过语义 smoke 后才可启用。在线 GRPO 可复用老板已验证的动态 LoRA 共享文件同步。
6. 首先用单个 TP8 实例做模型注册、HTTP 请求和输出语义 smoke；通过后可启用两个 TP8 副本并行 rollout。
7. 双副本评测时，task ID 到副本的分片固定；所有模型条件使用同一映射，避免副本差异混入模型差异。
8. base、chosen-SFT、continued-SFT、DPO、RPO、randomized-RPO 均在同一套 v21 internal heldout 上评测；最终结论另需全新 PI heldout。

结论：

- 训练与推理解耦是当前推荐架构，能消除训练/rollout 的 NPU 竞争，并把推理环境固定为单一版本。
- 该决策不改变 E-002 的权限门槛：5 号机 MindSpeed 26 训练仍需明确授权 privileged 容器。

### 2026-07-28：E-004 业界架构复核与 5→6 工程联调

业界架构复核：

- [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer) 和 [NeMo RL DPO](https://docs.nvidia.com/nemo/rl/latest/guides/dpo.html) 都将 DPO 定义为冻结偏好数据上的离线优化；训练期间不需要一台在线推理机持续生成样本。
- 因此本项目的 `5 号训练 + 6 号推理/评测` 是训练/评测解耦，不是两机共同执行一个 DPO optimizer。
- 真正的多节点训练应按 [PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed) 让多节点进入同一个 HCCL/NCCL distributed world，并由统一 launcher/scheduler 管理 rank、故障和 rendezvous。
- 在线 GRPO/PPO 才适合将 trainer 与 rollout worker 解耦；[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 和 [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html) 都采用框架管理的推理 worker、权重同步和调度，不使用个人电脑搬运 checkpoint。

本次实际更新：

- 新增 `scripts/publish_adapter_5_to_6.sh`，固化 5 号到 6 号的内网直传、相对路径 SHA256、`.incoming` 校验和原子发布；Windows 只发送控制命令。
- 5 号复用 chosen-SFT `checkpoint-75` 作为工程载荷，经内网直接传输到 6 号：
  - 传输字节：`233,694,155`
  - 观测吞吐：约 `155 MB/s`
  - tar SHA256：`6ca1a7cb3555104dab3d6607e1384cccef10303de9b5cc5ee661e6ee840f2ca8`
  - `adapter_config.json`：`66813c837dfbca64f0c86f190df7f0e9c011d52434e329a83ea0240f7957048a`
  - `adapter_model.safetensors`：`bcb7493f70519e89460ef9283616ee9e3cd6268dc0b3a7eabf0e975711bd42fa`
- 6 号完成逐包、逐文件校验并原子发布；未通过 Windows、HTTP 服务或第三方机器传输模型。
- 全程只复用用户自己的：
  - container：`llin-qwen36-grpo-pi-rollout-priv-host-0727`
  - image：`llin-vllm-ascend:grpo-pi-deps-20260727`
  未新建容器/镜像，未使用其他人的资源。
- 新增 `scripts/run_p001_vllm_smoke_inner.sh` 和固定请求 JSON，完成 TP8、vLLM 0.23.0、8K 工程启动：
  - `/v1/models` HTTP 200；
  - base 与 adapter 均成功注册；
  - base Chat Completions HTTP 200 且输出语义连贯。
- 静态 PEFT LoRA 路径发现两层兼容问题：
  1. `all-linear` 会命中 Qwen3.6 GDN `in_proj_ba`，触发 NPU Punica `hidden in should be smaller than hidden out`；
  2. 即使复用老板兼容补丁、过滤不支持张量，旧 chosen-SFT 和老板原生 GRPO checkpoint-20 都输出连续 `!`，虽然接口返回 200。
- 老板 GRPO 原运行的动态 rollout 轨迹正常，包含多轮 Bash 工具调用；20 步日志 reward 从 `0.25` 到 `0.36812499`。因此 checkpoint 与基座并非天然只会输出 `!`，失败边界在静态 PEFT LoRA serving 路径。
- `scripts/filter_qwen36_vllm_lora.py` 仅保留为诊断复现工具，不作为正式评测产物生成器。
- 将 `scripts/run_p001_megatron_inner.sh` 的 LoRA target 从 `all-linear` 修正为老板已验证的 `linear_qkv linear_fc1`，避免训练出当前 rollout 不支持的 GDN adapter。
- 所有冒烟 vLLM 进程已停止；6 号 16 张 NPU AICore 均回到 0。

结论：

- 多机控制面、5→6 服务器内网数据面、校验、原子发布、6 号 TP8 基座推理均已跑通。
- 静态 LoRA 虽“能加载、能返回 200”，但未通过语义门槛，不能记为完整闭环成功。
- P-001 的正式离线 DPO/RPO 采用“5 号训练 → 内网发布 → LoRA 合并 → 6 号冻结评测”；若后续做在线 GRPO，则直接复用老板已验证的动态 adapter 同步，不自建临时 HTTP/Windows 中转。
- 剩余工程门槛是：在 5 号仅使用 `llin-*` 资源启动可见 NPU 的训练容器，并完成 1-step checkpoint；随后在 6 号验证合并产物的语义输出。

### 2026-07-28：E-005 5 号训练 + 6 号 rollout 的同步在线 GRPO 单步闭环

架构决策：

- 初始阶段采用同步 on-policy GRPO：每一步先发布确定版本的 LoRA，等待所有 rollout worker 确认，再生成轨迹、计算 reward 和更新 trainer。
- 该选择与 [NeMo RL GRPO](https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html) 的同步 worker 设计一致；异步模式需要额外处理权重版本年龄和 off-policy 修正，留到同步基线稳定后再评估。[NeMo RL Async GRPO](https://docs.nvidia.com/nemo/rl/latest/guides/async-grpo.html) 也明确区分 rollout 与 trainer 的权重滞后。
- Windows 只发送 SSH 控制命令。模型、LoRA、轨迹和 checkpoint 均不经过 Windows。
- 5→6 数据面使用服务器内网 `192.168.202.5 → 192.168.202.4` 的 SSH/rsync；6 号 rollout 仅监听 `127.0.0.1:28220/28221`，5 号通过 SSH local forward 访问。
- 全程只使用或创建 `llin-*` 镜像和容器。

工程更新：

- 新增 `scripts/transfer_grpo_image_6_to_5.sh`，由 5 号直接拉取 6 号 trainer image，不经过 Windows。
- 新增 `scripts/create_p001_online_grpo_trainer_5.sh`，只接受 `llin-*` image；创建 privileged、host network/IPC 的 trainer，但不挂 Docker socket、不挂整块 `/data3`，只挂：
  - 必要 Ascend driver/管理文件（只读）；
  - Qwen3.6-27B base（只读）；
  - 本项目 reference（只读）；
  - 本次 `online_grpo` run 根目录（读写）。
- 新增 `scripts/cross_host_lora_sync_patch.py`：
  - trainer 将扁平 LoRA 保存为 `llin-ms-swift-flat-lora-v1`；
  - 记录 tensor 数、字节、SHA256 和唯一 `transfer_id`；
  - 等待服务器间 watcher ACK 后，才调用 rollout 的 adapter update endpoint；
  - 必须收到 `all_workers_loaded=true` 才允许生成轨迹。
- 新增 `scripts/watch_cross_host_lora_sync.sh`：
  - 5 号宿主直接 rsync 到 6 号 `.incoming`；
  - 在 6 号复算 SHA256/字节；
  - 发布为不可变 `versions/adapter-<transfer_id>-<sha>.pt`；
  - 原子替换 `adapter_flattened.pt` symlink；
  - 向 5 号写回版本化 ACK。
- 新增 `scripts/server_mode_no_local_vllm_patch.py`。5 号 trainer 只使用远端 HTTP client，不安装或执行第二套本地 vLLM/CANN；老板已有 shared-file patch 负责禁用 trainer 本地 HCCL communicator。
- 新增 `scripts/run_p001_online_grpo_train_inner.sh`、`scripts/run_p001_online_grpo_train_host_5.sh` 和 `scripts/start_p001_crosshost_grpo_5.sh`，固化：
  - trainer：`TP4 × PP1 × CP2 × SP`，8 张 NPU；
  - rollout：`TP4 × DP2`，8 张 NPU；
  - LoRA：rank 4、alpha 16、target `linear_qkv linear_fc1`；
  - `num_generations=8`、`generation_batch_size=8`；
  - smoke：总上下文 4096、completion budget 2048、1 step。
- 新增 `scripts/summarize_online_grpo_run.py`，在服务器就地汇总轨迹字段、reward/advantage、训练指标、checkpoint hash 和 safetensors finite 检查，不打印轨迹正文。

运行环境与兼容边界：

- 6 号老板 trainer image `llin-rl-grpo:pi-deps-20260727` 已由 5 号直拉，前后 image ID 完全一致：
  - `sha256:5b52febafc54df86cbaae7f6caa5e47da205cecba44217167d0f836754cf5c90`
  - `20,141,205,429` bytes
- 该 6 号 image 在 5 号直接运行时出现 CANN/driver ABI undefined symbol，不能复用为 5 号正式 trainer。
- 5 号新构建的 `llin-qwen36-p001-megatron:20260728` 可见 8 卡，但其 `triton-ascend 3.2.0` 与 5 号 CANN header 不兼容；失败镜像和容器保留审计，未删除。
- 最终复用 5 号已经完成 DPO 训练的自有基座 `llin-rl-dpo-p2-base:20260707`。它的 MindSpeed 导入与 GRPO Python 依赖均通过，最终 trainer container 为：
  - `llin-qwen36-grpo-trainer-m05-p001-dpo-base`
- 6 号 rollout container 为：
  - `llin-qwen36-grpo-pi-rollout-priv-host-0727`

失败迭代：

| run | 失败边界 | 处理 |
|---|---|---|
| r1 | 5 号新镜像找不到 `ccec`，补 PATH 后又暴露 Triton/CANN header 不兼容 | 改用 5 号已验证的自有 DPO 基座 |
| r2 | server mode 仍被 Swift trainer 的“本地 vLLM 必须存在”检查阻止 | 增加 trainer-only remote-vLLM patch |
| r3 | 通用 RLHF guard 已覆盖，但 Megatron trainer 还有第二层同名 guard | 对两个模块都做显式 server-mode patch 和模块级断言 |
| r4 | 成功 | 完成同步、8 条轨迹、reward、反向更新和 checkpoint |

r4 工程证据：

- run：`p001_crosshost_grpo_1step_20260728_r4`
- 初始 LoRA：
  - tensor 数：`408`
  - bytes：`28,186,699`
  - SHA256：`224c2eb37844d6dbe8a260c7b72de6270f49691b1182ec050f83b210757a725e`
  - 5 号 request 到 6 号 ACK：约 2 秒
  - 5、6 号复算 SHA256 完全一致
  - 6 号 8 个 worker 均写出 `adapter_loaded`；server 返回 `all_workers_loaded=true`
- rollout：
  - `8/8` 轨迹完成，`/infer/` HTTP 200
  - 单次 8 样本生成约 42 秒
  - `completions.jsonl`：`74,810` bytes
  - SHA256：`da54df7488c603e7f19a899f4b85ac3e773ea5358e2e563c839f847f3feb6a34`
- trainer：
  - exit code：`0`
  - step：`1/1`
  - loss：`4.62e-06`
  - KL：`1.1551e-4`
  - grad norm：`1.08709e-3`
  - learning rate：`1e-6`
  - trainer step：约 `261.48s`
  - NPU memory：约 `16.02GiB`
- checkpoint：
  - `checkpoint-1/latest_checkpointed_iteration.txt == 1`
  - 同时保存 8-rank Megatron distributed checkpoint 和 LoRA safetensors
  - `adapter_model.safetensors`：`28,181,384` bytes
  - SHA256：`f2341828039feb34c719f1c2d14832ddf8079d74f248843a17789f29e42f5936`
  - 408 个 adapter tensor 全部可读且 finite

质量结论：

- 8 条 `PiAgentTrajectoryORM` reward 均为 `0.3000000119`，reward std 为 `0`。
- 8 条 advantage 均为 `0`，`frac_reward_zero_std=1.0`。
- 非零 loss/grad 说明完整训练代码路径执行了，但本 batch 没有 GRPO 组内排序信号；本实验不得记为“模型效果提升”。
- 当前工程已经足够进入 reward-signal pilot，但还不足以直接长跑在线 GRPO。

下一步：

1. 固定当前 r4 工程栈，先对 20 个唯一 prompt 各采样 8 条，仅做 rollout/verifier audit，不更新模型。
2. 逐 prompt 统计 reward 分布、成功类型、工具调用失败类型和 completion budget 命中率；将 `reward_std=0` 的原因区分为任务过易、任务过难、verifier 过粗和采样同质化。
3. 优先修 verifier 的组内区分度，再调温度/采样；不通过单纯扩大训练步数掩盖零 advantage。
4. 选择有非零组内方差且 verifier 可信的 prompt 组成小型 pilot；加入固定 base/chosen-SFT/continued-SFT/verified-RPO 对照。
5. pilot 通过后再考虑异步 GRPO；异步必须记录 weight version/age，并设置最大可接受陈旧度，不能直接复用同步结果做质量结论。

资源状态：

- r4 完成后已停止 5 号 trainer、SSH tunnel 和 LoRA watcher。
- 6 号 rollout container 已重启回空闲主进程，28220/28221 无监听。
- 5 号意外重新启动的已完成旧 DPO 容器 `llin-rl-dpo-p2-formal-0-3` 已再次停止。
- 5、6 号 `npu-smi` 均显示无 NPU 进程；所有失败 run、成功 run、镜像、容器定义和 checkpoint 均保留。

### 2026-07-28：E-006 在线 GRPO reward-signal audit（用户停止，保留 17/20 组）

目标与约束：

- 固定 E-005 r4 的初始策略，仅在 6 号做在线 rollout/verifier audit；不启动 5 号 trainer、不执行 optimizer step、不更新权重。
- 计划对 20 个唯一 prompt 各采样 8 条，以组内 reward 方差判断是否存在 GRPO 排序信号。
- Windows 仍只发送 SSH 控制命令；轨迹正文、模型和 checkpoint 均留在服务器。
- 全程只使用自有容器 `llin-qwen36-grpo-pi-rollout-priv-host-0727` 和 `llin-*` image。

工程更新：

- 新增 `scripts/audit_online_grpo_reward_signal.py`：
  - 按 prompt 原子写入 8 条完整 group，支持断点跳过；
  - 将数据集元数据通过 rollout API 的 `data_dict` 透传；
  - 记录 reward 分量、停止原因、工具失败、token budget 和轨迹 SHA256；
  - 支持 `--start-prompt` / `--end-prompt` 分片，只有完整 20 组才写正式 summary。
- 新增 `scripts/load_shared_lora_adapter.py`，在采样前校验共享 LoRA 的 SHA256，并要求 8 个 rollout worker 全部报告 `adapter_loaded`。
- 新增 `scripts/run_p001_reward_signal_audit_6.sh`，固化自有容器检查、r4 LoRA 精确复制、服务健康检查、脱敏环境证据和退出时资源清理。
- 新增 `scripts/summarize_partial_reward_audit.py`，只汇总已完成 group 的统计量与文件哈希，不输出轨迹正文。
- 新增 `tests/test_reward_signal_audit.py`，覆盖 `data_dict` 透传、零/非零方差诊断和“audit 不授权训练”的门禁。

运行记录：

| run | 结果 | 说明 |
|---|---|---|
| `p001_reward_signal_audit_20x8_20260728_r1` | 失败并自动释放资源 | rollout 总上下文为 4096；首个 prompt 中一条输入已达 4460 token，剩余 completion budget 变成负数并触发 HTTP 500。未生成有效 group，未更新权重。 |
| `p001_reward_signal_audit_20x8_20260728_r2` | 用户停止，保留 17/20 组 | 将总上下文提高到 8192，同时保持 completion budget 2048；完成 prompt 0–16，共 136 条轨迹。未更新权重。 |

r2 冻结证据：

- 策略沿用 E-005 r4 LoRA：
  - bytes：`28,186,699`
  - tensor 数：`408`
  - SHA256：`224c2eb37844d6dbe8a260c7b72de6270f49691b1182ec050f83b210757a725e`
  - 8 个 worker 全部确认加载，`all_workers_loaded=true`
- 数据 SHA256：`e819848b0cdde6f69bdfb08537060e02bcf6d95ea64a8f7d212539517dfc6b57`
- verifier manifest SHA256：`e4269e118605c24773cfe749d479cc8cbdb637dd23fd277895d18e70233652ee`
- reward plugin SHA256：`23f27019d93f31af4a30592a3c29522bc18201cfbb700538d50d733399cc00bd`
- 为缩短 wall time，主进程完成 prompt 0–9，额外分片完成 prompt 10–16；在 prompt 17 写文件前主动停止额外分片，避免两个进程竞争同一 group 文件。
- 用户发出停止指令后立即停止剩余采样；prompt 17–19 没有被记为完整 group，也没有伪造 20/20 summary。

17 组脱敏汇总：

| 指标 | 结果 |
|---|---:|
| 完整 prompt group | 17 |
| 完整轨迹 | 136 |
| reward 均值 / 中位数 | 0.3346 / 0.3 |
| reward 计数 | `0.0: 19`、`0.3: 90`、`0.5: 17`、`1.0: 10` |
| 非零组内方差 | 16/17 |
| 零方差 prompt | index 16，8 条均为 `0.3` |
| 命中总 token 上限 | 91/136（66.9%） |
| 停止原因 | `total_token_limit: 91`、`final_answer: 44`、`max_turns: 1` |
| 单条耗时 | 中位数 66.70s，均值 120.70s，最大 582.29s |
| 工具调用次数 | 中位数 4，均值 4.95，最大 15 |

信号判断：

- prompt index 2、10、11 同时出现 `0.0` 和 `1.0`，是首批值得人工核验的高区分度候选。
- index 5、6、7、8、14 的方差较弱；index 16 完全零方差。它们需要区分 verifier 过粗、任务难度、采样同质化和 token 截断，不宜直接混入训练。
- 136 条中 `queried_required_tables=true` 为 38，`gold_evidence=true` 为 12；说明当前高分信号稀疏。
- 工具失败事件以 `command_not_found: 137`、`policy_blocked: 27`、`execution_error: 26`、`missing_file: 22` 为主。一次轨迹可包含多个失败事件，因此这些计数不能直接当作轨迹失败率。
- rollout 返回的 `generated_tokens` 字段在 136 条中均为 0，而 `tool_response_tokens` 中位数为 1984；该字段口径需要在正式训练前核实，当前只使用停止原因和总 budget 命中判断截断。
- 本次结果证明 17 个已完成 prompt 中存在 GRPO 组内排序信号，但不证明 verifier 已可信，更不证明模型效果提升；`quality_claims_allowed=false`、`policy_update_performed=false`。

停止与资源状态：

- 用户停止后，6 号 rollout container 已重启回只运行 `sleep infinity`，没有 rollout 服务。
- 5 号 trainer container 保持退出状态。
- 最终复核 5、6 号 `npu-smi`：所有可见 NPU 的 AICore 为 0%，均显示 `No running processes found`。
- 17 个完整 group 和脱敏 `partial_summary.json` 留在 6 号服务器，未传到 Windows；后续不会自动续跑。

下一次只有在用户明确恢复实验后才执行：

1. 人工复核 index 2、10、11 的 reward 分量与轨迹正确性，并抽查 0.3 大量聚集是否为合理 verifier 判定。
2. 修正或确认 token 统计口径，降低 66.9% 的 budget hit，再冻结候选 prompt。
3. 只对 verifier 可信且组内方差非零的 prompt 做小型在线 GRPO pilot，并保留 base/chosen-SFT/continued-SFT 对照；不得由本次 audit 自动授权训练。

### 2026-07-29：E-007 PI 轨迹 GRPO v2 契约、反事实审计与无训练 smoke

方法决策：

- 工具执行结果是环境 observation：必须保留在后续上下文中，但不属于策略动作，不进入 policy loss。
- assistant 内容和 tool call 是策略 action：进入 loss；prompt、system、user 和 tool response 全部显式 mask。
- 默认不采用“每一步都奖励”的稠密过程级 GRPO。当前没有独立、校准过的 step verifier，直接按调用次数、命令成功或中间文本给分会奖励绕路、重复调用和 verifier hack。
- v2 使用保守混合 reward：
  - `1.0`：真实 `final_answer`，且 safe、tool protocol 有效、工具成功、查询必需表、命中 gold evidence；
  - `0.2`：真实 `final_answer`，满足以上前置条件但尚未命中 gold evidence，作为可验证进展；
  - `0.0`：截断、仅安全、仅有文本答案、一般工具成功、未查询必需表或其他情况。
- 工具结果可以作为 verifier 判断任务结果的证据，但工具结果 token 本身不获得梯度，也不因为“被执行出来”自动得分。
- 该选择与 [Search-R1](https://arxiv.org/abs/2503.09516) 的 retrieval observation masking + outcome reward、[ReTool](https://arxiv.org/abs/2504.11536) 的结果导向工具强化学习一致；细粒度 credit assignment 留到有独立 step verifier 后再评估，如 [Agent Lightning](https://arxiv.org/abs/2508.03680) 所讨论的 agent credit assignment。

现有实现审计：

- 真实 ms-swift Qwen3.6 模板验证通过：
  - 总 token：340
  - 可训练策略动作 token：50
  - 被 mask 的 prompt/observation token：290
  - assistant action、tool call、final action 均可训练；
  - user observation、tool result 均存在于上下文但不进入 loss。
- 发现旧调度器的两个证据缺陷：
  1. HTTP rollout 返回的 `choice.token_ids` 为空，旧 `generated_tokens` 因而在 136 条中全部错误记录为 0；
  2. 旧 verifier 用“最后存在 assistant 内容”代替真实终止原因，导致 136 条都被记为 `has_final_answer=true`，但实际只有 44 条以 `final_answer` 结束，另外 92 条已截断。
- v2 在引擎不返回 token IDs 时使用 tokenizer 回退计数，并分别记录：
  - `policy_action_tokens`
  - `engine_reported_policy_tokens`
  - `fallback_policy_tokens`
  - `observation_tokens`
- 轨迹预算拆分为总预算、策略保留预算、累计 observation 上限和单次工具结果上限；工具输出仍可进入上下文，但不能吞掉全部策略预算。
- ms-swift 当前 HTTP response schema 不接受布尔型 message `loss`。v2 改为 schema 可接受、模板真值语义等价的字符串标记；修正后真实模板仍得到完全相同的 `50 trainable / 290 masked`。

136 条历史轨迹的 CPU-only 反事实重放：

| reward 契约 | reward 分布 | 非零组内方差 |
|---|---|---:|
| 旧 v1 | `0:19 / 0.3:90 / 0.5:17 / 1:10` | 16/17 |
| 纯终局 outcome | `0:126 / 1:10` | 4/17 |
| v2 保守混合 | `0:115 / 0.2:11 / 1:10` | 12/17 |

- 真实终局答案：44/136；截断：92/136。
- outcome success：10；verified progress：11。
- 纯终局 reward 对当前数据过稀疏；旧 v1 又大量奖励 `0.3` 的浅层条件。v2 保留 12/17 组方差，同时删除“安全调用/任意答案即得分”。
- 失败责任脱敏计数：
  - agent command-not-found：137
  - agent execution error：26
  - agent missing resource：22
  - agent policy blocked：27
  - ambiguous other：11
- 反事实摘要只保留在 6 号服务器，SHA256：`432db232d7ef3455036c91af96db0fda8ac06147b9468f149cd90203930492d3`；没有执行 optimizer 或权重更新。

工程更新：

- 新增 `scripts/pi_trajectory_contract.py`：统一 action mask、observation budget、失败责任和 v2 reward 决策。
- 新增 `scripts/pi_agent_grpo_v2_plugin.py`：v2 scheduler 与 ORM。
- 新增 `scripts/audit_pi_trajectory_v2.py`：历史轨迹 CPU-only 反事实审计，只输出脱敏聚合。
- 新增 `scripts/verify_pi_action_loss_contract.py`：用真实 tokenizer/template 验证 observation 零 loss。
- 新增 `scripts/run_qwen36_grpo_pi_rollout_v2_inner.sh` 和 `scripts/run_p001_trajectory_v2_smoke_6.sh`。
- 在线 trainer 默认切换到 `pi_agent_scheduler_v2 + pi_agent_trajectory_v2 + loss_scale=default`，但本次没有启动 trainer。
- reward audit 支持 v1/v2 契约、prompt 范围、可变模型上下文/轨迹预算和部分安全汇总。
- 安全汇总新增 action/observation token、真实终止、v2 判定和失败责任统计；不输出轨迹正文。
- 新增 9 项测试，覆盖 observation mask、截断不得伪装终局成功、混合奖励、失败责任、预算保护和 audit 门禁；全部通过。
- 新增 `.gitattributes`，固定 Markdown、Python、shell 文件为 LF，避免 Windows checkout 破坏服务器脚本。

无训练 smoke：

| run | 参数 | 结果 |
|---|---|---|
| `p001_trajectory_v2_prompt0_smoke_20260729_r1` | prompt 0，8 条，2048 budget | 轨迹生成后被 ms-swift HTTP schema 拒绝布尔 `loss`；无完整 group、无训练、自动释放 NPU。 |
| `p001_trajectory_v2_prompt0_smoke_20260729_r2` | prompt 0，8 条，2048 budget | 8/8 完整返回，接口修复通过；reward 全 0；`length:3 / observation_token_limit:5`，真实终局 0。证明契约可运行，但预算不足。 |
| `p001_trajectory_v2_prompt2_budget3072_smoke_20260729_r3` | prompt 2，8 条，model 12288 / trajectory 3072 | 8/8 完整返回；reward `0:4 / 0.2:1 / 1:3`，均值 0.4，组内标准差 0.4690；真实终局 4，observation 截断 4。 |

r3 脱敏证据：

- outcome success：3；verified progress：1；组内存在三档 reward。
- action token：中位数 1288.5，均值 1260，范围 925–1492。
- observation token：中位数 976.5，均值 925.4，范围 727–1024。
- 引擎原生 token 计数仍为 0，但 tokenizer 回退计数覆盖 8/8，旧的全零统计问题已修复。
- 完整 group SHA256：`e83ae3dff62bd34ac7dc3ee0f52ea328bafaa1f5406d83573bf21a92f335171c`。
- `quality_claims_allowed=false`、`policy_update_performed=false`；不能把该结果解释为模型效果已经提升。

当前门禁与下一步：

- v2 action/observation 契约、reward 契约、多机 rollout 接口和脱敏审计已经工程跑通。
- prompt 2 证明混合 reward 可以产生有效组内排序信号，但单题不足以授权训练，且 4/8 仍在 observation budget 处截断。
- 暂不启动 5 号 optimizer。下一步先实现 observation budget 耗尽后的受控 finalization 回合，再在多个历史高区分度 prompt 上复验终局率、reward 方差和 verifier 正确性。
- 通过后只运行 1-step v2 GRPO，检查 advantage 非零、loss/grad finite、checkpoint 和 5→6 权重同步；仍不自动授权长跑。

资源状态：

- 所有 v2 smoke 都只在 6 号执行 rollout，没有让 Windows 参与模型、轨迹或 checkpoint 数据面。
- 只使用自有容器 `llin-qwen36-grpo-pi-rollout-priv-host-0727` 和 `llin-*` image。
- 5 号 trainer `llin-qwen36-grpo-trainer-m05-p001-dpo-base` 保持停止。
- r3 完成后 6 号 rollout container 回到只运行 `sleep`；5、6 号均无 NPU 进程。

### 2026-07-29：E-008 两台全机 40K 轨迹 GRPO 单步闭环

目标与边界：

- 5 号机专职训练，6 号机专职在线 rollout；Windows 仍只发送 SSH 控制命令，不承载模型、轨迹、checkpoint 或两机数据中转。
- 两台机器各有 8 块双芯片板卡，即各 16 张计算 NPU；此前自有 GRPO 容器只暴露 8 张，本次扩展为两台各使用完整 16 张。
- 只创建和使用 `llin-*` 容器及镜像；旧自有容器、镜像、run 和 checkpoint 均保留，没有覆盖或删除。
- 本次只授权工程单步，不自动扩展为长跑，也不以单题训练结果宣称模型质量提升。

上下文选择：

- 训练拓扑固定为 `16 ranks / TP8 × PP1 × CP2 × SP`；rollout 固定为 `TP8 × DP2`。
- 选择 `max_model_len=40960`、轨迹总预算 `39936`：
  - 40K 已在长上下文实验中完成 worst20 多样样本、20/20 step 和严格 checkpoint 恢复；
  - 48K 只有固定 worst1 压力覆盖，不能代表多样样本稳定性；
  - 100K 的峰值曾到 `63.997/64 GiB`，只剩约 3 MiB，不具备可恢复余量。
- 使用真实 tokenizer/template 在 6 号服务器内审计 20 个 prompt，不输出 prompt 正文：
  - token 范围：765–801；
  - 最大 prompt：801 token；
  - `801 + 39936 = 40737`，相对 40960 保留 223 token 的模板/边界余量。
- 40K rollout 预算拆分：
  - 策略保留：16384；
  - 累计 observation：8192；
  - 单次 observation：2048；
  - 受控 finalization 保留：2048；
  - 单回合策略生成上限：4096。

工程更新：

- 新增全机自有容器创建脚本：
  - `scripts/create_p001_maxctx_trainer_5.sh`
  - `scripts/create_p001_maxctx_rollout_6.sh`
- 新增 `scripts/run_p001_online_grpo_rollout_host_6.sh`：提供常驻 rollout、就绪标记、NPU 时序记录和退出清理。
- 新增 `scripts/audit_pi_prompt_token_lengths.py`：只输出 prompt token 长度和任务标识，不输出 prompt 正文。
- 新增 `scripts/select_pi_prompt_subset.py`：在服务器内按索引选择训练行并输出行数/SHA256，不让数据经过 Windows。
- trainer 的 rank、TP、CP、PP、数据集、generation batch、global batch 均改为显式可配置，并增加拓扑乘积与批大小前置校验。
- rollout 的 TP/DP、上下文和轨迹预算改为显式可配置，并校验 `TP × DP` 等于可见计算 NPU 数。
- v2 scheduler 新增受控 finalization：
  - observation 预算耗尽后追加一个被 mask 的环境指令；
  - finalization 回合禁用工具，只训练 assistant 最终动作；
  - 单独记录 finalization 是否触发、是否成功及失败停止原因；
  - observation 永远不能消耗 finalization 保留预算。
- 新增单回合策略上限 4096，防止一次 assistant 生成独占整个 39936 token 的轨迹预算；轨迹累计上限仍保持 39936。
- 脱敏汇总新增 finalization 计数；reward audit 将 finalization 长度/工具失败正确归为截断。
- 本地 11 项相关测试全部通过，Python 编译、shell 语法检查和 `git diff --check` 均通过。

容器与数据：

| 机器 | 自有容器 | 自有镜像 | 可见计算 NPU |
|---|---|---|---:|
| 5 号 trainer | `llin-qwen36-grpo-trainer-m05-p001-maxctx-0729` | `llin-rl-dpo-p2-base:20260707` | 16 |
| 6 号 rollout | `llin-qwen36-grpo-pi-rollout-m06-maxctx-0729` | `llin-vllm-ascend:grpo-pi-deps-20260727` | 16 |

- 单题训练集在 5 号服务器内由原始 20 题数据选出 prompt index 2：
  - 行数：1；
  - SHA256：`dfacdc4c6ea16cf8d13669539591f0877e53bab4539f10c50d31a8904a77dbfd`；
  - 数据正文没有经过 Windows。

40K rollout 门禁：

| run | 结果 | 说明 |
|---|---|---|
| `p001_trajectory_v2_prompt2_maxctx40k_finalization_smoke_20260729_r4` | 主动停止，无完整 group | 初版允许单个 assistant 回合最多使用全部 39936 token；确认请求会被单回合长生成拖住后按精确 PID 停止，未训练并自动释放 NPU。由此加入 4096 单回合策略上限。 |
| `p001_trajectory_v2_prompt2_maxctx40k_finalization_smoke_20260729_r5` | 通过 | 8/8 完整轨迹，全部自然 `final_answer`，无预算截断、无 OOM；reward `0×2 / 0.2×1 / 1×5`，均值 0.65、组内标准差 0.4555。 |

r5 额外证据：

- 终局答案：8/8；budget hit：0/8。
- 策略 action token：1281–1832，均值 1470.6。
- observation token：937–5674，均值 2407.4。
- finalization 触发/成功：0/0；原因不是功能失效，而是 40K 预算已让 8 条轨迹全部自然结束。
- 初始老板 LoRA 精确加载：
  - bytes：`28,186,699`
  - tensor 数：`408`
  - SHA256：`224c2eb37844d6dbe8a260c7b72de6270f49691b1182ec050f83b210757a725e`

全机单步联调：

| run | 结果 | 说明 |
|---|---|---|
| `p001_crosshost_grpo_v2_maxctx40k_1step_20260729_r1` | 参数校验失败，无训练 | 16-rank world 下 `generation_batch_size=8` 导致 per-device generation batch 为 0。未加载模型、未 rollout、未更新权重；清理两机进程和 NPU。 |
| `p001_crosshost_grpo_v2_maxctx40k_1step_20260729_r2` | 参数校验失败，无训练 | 将 generation batch 提高到 16 后，框架要求它等于 `global_batch_size × steps_per_generation`；旧 global batch 仍为 8。未加载模型、未 rollout、未更新权重；清理两机进程和 NPU。 |
| `p001_crosshost_grpo_v2_maxctx40k_1step_20260729_r3` | 工程通过、参数未变化 | 固定 `generation/global batch=16/16`、`num_generations=8`、`steps_per_generation=1`；完成 16 条轨迹、反向和 checkpoint-1。后续 E-009 发现 1-step warmup 令本步实际学习率为 0，LoRA-B 未更新。 |

r3 同步与训练证据：

- 5→6 初始 LoRA 同步：
  - bytes：`28,186,699`
  - tensor 数：`408`
  - SHA256：`80aa906851ddd161fcd4e63d5336bbc90354e06dde445153ce67ebd4fa848556`
  - 6 号版本目录、原子发布、ack 哈希一致；全部 rollout worker 报告加载成功。
- 16 条轨迹分成两个 8 条 GRPO group：
  - reward：`0×3 / 0.2×2 / 1×11`
  - reward 均值：`0.7125`
  - reward std：`0.4535`
  - `frac_reward_zero_std=0`
  - advantage：16/16 非零且 finite，范围 `[-1.84465, 0.71858]`
- 单步数值：
  - loss：`4.315e-05`
  - KL：`8.028e-05`
  - grad norm：`0.08107854`
  - learning rate：`1e-06`
  - 以上数值全部 finite，无 OOM、Traceback 或 RuntimeError。
- 首次 40K/TP8×CP2 shape 包含 Ascend Triton 编译，完整单步耗时 17 分 11 秒；后续同 shape 可复用编译缓存。
- checkpoint：
  - `latest_checkpointed_iteration.txt == 1`
  - LoRA `adapter_model.safetensors`：`28,181,384` bytes
  - tensor 数：`408`
  - 所有 tensor finite：`true`
  - SHA256：`f2341828039feb34c719f1c2d14832ddf8079d74f248843a17789f29e42f5936`
  - 同时保存完整 16-rank Megatron distributed checkpoint。
- 脱敏 completions 文件只留在 5 号服务器：
  - 记录批数：1，包含 16 条 completion；
  - SHA256：`64278ca231ed4f4a26b0ed75484601eb609b1b1e41ee26c5d6ddae69bdb395f8`；
  - 未将轨迹正文复制到 Windows 或 GitHub。

结论与后续门禁（经 E-009 追溯更正）：

- 已证明两台各 16 张计算 NPU、40K 上下文、服务器内网权重同步、在线多轮工具轨迹、v2 reward、反向计算和 checkpoint 的完整工程链路。
- r3 的 Adam moment 非零，但 LoRA-B、FP32 master B 和 HF adapter 均未变化；不能再将 r3 单独作为“参数已更新”的证据。真实参数更新由 E-009 的 warmup=0 修正版单步证明。
- 已证明本次单题的两个 group 都有非零 reward 方差和 advantage；不同于旧半机 smoke 的全零 advantage。
- 尚未证明模型效果提升：当前只有一个 prompt，且因为全机 batch 对齐被重复成两个 group；不得把单步 reward 或训练 loss 当成 heldout 改善。
- 长跑前仍需冻结多个 verifier 可信且组内方差非零的 prompt，保留 chosen-SFT / continued-SFT / verified-RPO 对照，并运行独立 heldout 评测。

最终资源状态：

- 5、6 号的训练、rollout、SSH tunnel 和 LoRA watcher 均已按精确 PID 停止。
- 28220/28221/29693 无监听；两机 `npu-smi` 均显示所有 NPU `No running processes found`。
- 两个新自有容器仍为 running，但内部都只有 `sleep infinity`，不占用 NPU。
- r1、r2 失败证据，r3 成功证据、checkpoint、哈希和日志全部保留；没有自动启动下一步或长跑。

### 2026-07-29：E-009 多提示 40K 审计、首步 warmup 修复与真实参数更新

目标与边界：

- 在长跑前用多个 verifier 可信、组内 reward 方差非零的 prompt 复验 v2 轨迹质量。
- 首个多提示 batch 固定为两个 prompt × 每组 8 条 = 16 条，只做 1 个 optimizer step。
- Windows 只发送 SSH 控制命令；数据集、轨迹、LoRA、checkpoint 和 5→6 同步均留在服务器。
- 只使用自有 `llin-*` 容器/镜像；不启动长跑，不以单步训练指标宣称模型效果提升。

协议与审计更新：

- `audit_online_grpo_reward_signal.py` 新增非连续原始 prompt index 选择，避免重排后丢失任务身份。
- v2 scheduler 修复受控 finalization 的消息边界：工具观察后不再追加破坏模板角色序列的独立 user message，而是将被 mask 的最终回答指令合并到最后一条 observation；工具输出仍不参与 loss。
- 本地轨迹/审计回归测试 14/14 通过；5/6 号部署的 scheduler 和 contract SHA256 分别一致为：
  - scheduler：`8a18acf09c8a83bc1128be2da4ca64952a3db4aeba4f5b191c14f093ec7947f6`
  - contract：`390751f6b8d56889a08ad341add822c0d509773a86467348be936cb08e8c3afc`

40K 无训练审计：

| run | prompt | 结果 |
|---|---:|---|
| `p001_trajectory_v2_prompts2_10_11_maxctx40k_audit_20260729_r1` | 2/10/11 | prompt 2 完成后，prompt 10 触发 `response_role: user` 模板断言；无训练、自动清理。prompt 2 为 `1×5 / 0×3`，8/8 终局。 |
| `p001_trajectory_v2_prompt10_maxctx40k_finalization_fix_20260729_r2` | 10 | 8/8 HTTP 成功；`1×5 / 0×3`，7/8 终局、1/8 max-turn 截断。 |
| `p001_trajectory_v2_prompt11_maxctx40k_finalization_fix_20260729_r3` | 11 | 8/8 HTTP 成功；`1×5 / 0×3`，6/8 终局、2/8 max-turn 截断。 |

- 三个 prompt 都有非零组内 reward 方差；合计 24 条中 21 条终局，终局率 87.5%。
- 24/24 工具协议有效且有成功工具调用。
- prompt 2/10 的 required-table 覆盖均为 8/8；prompt 11 只有 5/8，故首个训练 batch 选择 2+10，将 11 保留为后续难例/课程组。
- 两行训练集在 5 号服务器内生成，SHA256：`80a642c8a9ecf3958805f53dbb8eed1a1e205eacfda77a4d088f89cc85e3d521`；未输出数据正文。

多提示单步与零学习率追溯：

| run | 结果 | 说明 |
|---|---|---|
| `p001_crosshost_grpo_v2_prompts2_10_maxctx40k_1step_20260729_r1` | 反向通过、参数未变化 | 16 条轨迹、两组 reward 方差非零、16/16 advantage finite；但沿用 `lr_warmup_fraction=0.03`，1-step 实际学习率为 0。DCP model/master LoRA-B 仍全零，HF adapter 哈希仍为初始化值 `f2341828...`。 |
| `p001_crosshost_grpo_v2_prompts2_10_maxctx40k_warmup0_1step_20260729_r2` | 真实参数更新通过 | 唯一训练差异是显式 `lr_warmup_fraction=0`；完成 16 条轨迹、反向、optimizer step 和 checkpoint-1。 |

warmup=0 修正版证据：

- reward：`0×4 / 0.2×3 / 1×9`，均值 `0.6000`，总体 reward std `0.3618`。
- prompt 2：`1×7 / 0.2×1`，组内标准差 `0.2646`。
- prompt 10：`1×2 / 0.2×2 / 0×4`，组内标准差 `0.4123`。
- advantage：16/16 非零且 finite，范围 `[-2.4740, 1.5877]`。
- loss `1.1174e-4`、KL `7.754e-5`、grad norm `0.06853`，全部 finite；无 OOM、HTTP 500、模板断言或 RuntimeError。
- checkpoint：
  - `latest_checkpointed_iteration.txt == 1`
  - HF adapter：408 tensors、全部 finite、SHA256 `eddb8468ac5874eeef7931adf5bb803573867bd1ecab04faedabb5449ff64cf8`
  - HF LoRA-B：204 个 tensor 中 176 个非零
  - DCP BF16 model / FP32 master：108 个 fused B 中均有 80 个非零
  - BF16 model 与 FP32 master 最大绝对差：`2.98e-08`
- completions 只留在 5 号服务器，SHA256：`45e141c7f34e5b149a3e1df8e6ece0b3ba16752252cce78006a1c37e028fb102`；未复制轨迹正文。

结论与后续门禁：

- 现已证明 40K、两台各 16 NPU、两个真实 prompt、服务器内网权重同步、有效组内 advantage、实际 optimizer 参数变化和可用 HF adapter 的完整闭环。
- 1-step smoke 必须显式关闭 warmup，或至少运行超过 warmup 区间；今后“训练成功”门禁必须包含 LoRA-B 非零和更新前后权重哈希/数值差，不再只看 grad norm、iteration 或 checkpoint 存在。
- 尚未证明 heldout 效果提升；长跑仍未授权。下一阶段应冻结 matched 起点与独立 heldout，对 continued-SFT / 普通 DPO / verified RPO / trajectory-GRPO 做同预算比较。

最终资源状态：

- 5、6 号本次训练、rollout、SSH tunnel 和 LoRA watcher 均按精确 PID 停止。
- 28220/28221/29694 无监听；两机 NPU 均显示 `No running processes found`。
- 两个自有全机容器保持 running，但内部只有 `sleep infinity`，不占用 NPU。
