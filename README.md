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

多机工程方面，5→6 内网 checkpoint 直传、校验和原子发布已通过；6 号基座推理已通过。当前仍不能宣告完整训练/评测闭环通过：vLLM-Ascend 0.23 的静态 PEFT LoRA 路径虽能注册 adapter，但在两个真实 checkpoint 上均产生重复感叹号，不能作为质量验收路径。P-001 将改用训练后 LoRA 合并模型进行离线评测，或复用老板已验证的动态 LoRA 同步链路。

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

状态：`data-frozen / multihost-data-plane-passed / training-runtime-gated`

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
