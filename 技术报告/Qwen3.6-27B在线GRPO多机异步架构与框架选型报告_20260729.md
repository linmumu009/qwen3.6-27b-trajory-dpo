# Qwen3.6-27B 在线 GRPO 多机异步架构与框架选型报告

**报告日期：** 2026-07-29  
**适用对象：** 在线强化学习、训练平台、推理平台与 PI Agent 工程团队  
**决策范围：** 在现有 Ascend NPU、Qwen3.6-27B、40K 多轮工具轨迹和跨机 LoRA 同步基础上，评估是否以及如何从同步 GRPO 演进到多 rollout 节点的全异步 Agentic GRPO  
**当前状态：** 架构调研完成；同步基线已验证；异步 PoC 尚未启动；不授权直接长跑

---

## 一、技术摘要

### 1.1 结论

本项目已经完成一条可信的同步在线 GRPO 工程基线：

- 5 号机使用 16 张计算 NPU 训练；
- 6 号机使用 16 张计算 NPU、`TP8 × DP2` 提供两个 rollout 副本；
- 40K 上下文、多轮 PI 工具轨迹；
- 两个真实 prompt，每个 prompt 生成 8 条轨迹；
- 16/16 advantage 非零且 finite；
- LoRA、Megatron BF16 权重和 FP32 optimizer master 均发生真实、finite 的参数更新；
- LoRA 通过服务器内网、SHA256、不可变版本目录和原子 symlink 发布；
- Windows 只承担 SSH 控制，不进入模型、轨迹与 checkpoint 数据面。

因此，下一阶段的核心问题不再是“GRPO 能否运行”，而是：

> 如何让多个 rollout 节点持续生成轨迹，由全局调度器按完整 GRPO group 聚合，训练端凑够一个 optimizer batch 后立即训练，同时允许 rollout 继续运行，并正确处理权重版本、长尾轨迹和 off-policy 偏差。

该目标在业界已有成熟方向，通常称为：

> **训推分离的全异步 Agentic GRPO + 全局 Group Buffer + 有界陈旧度 + Partial Rollout。**

在当前 Ascend 环境下，推荐顺序为：

1. **第一验证候选：veRL-NPU。** 它的 Fully Async 架构、全局 MessageQueue、陈旧度控制、Partial Rollout、rollout logprob 与 Importance Sampling 最接近目标，同时拥有当前最完整的 Ascend 快速启动和训练/推理后端组合。
2. **第二验证候选：AReaL Ascend。** 它在算法和 Agent 工作流层面最完整，原生全异步，支持跨策略版本的部分轨迹及 Decoupled PPO；但需要额外验证 Ascend 分支与当前 Qwen3.6、40K、LoRA、Megatron 组合的交叉兼容性。
3. **第三候选：阿里 ROLL。** 已支持 Ascend、Agent 环境级异步 rollout 和异步训练，工程可用性较高；但公开资料中对跨版本 Partial Rollout 和样本级陈旧度控制的描述不如前两者完整。
4. **MindSpeed-RL 只作为实现参考。** 它在 Ascend 上实现的 TransferDock 和“完成数达到 GBS 即训练”与本项目设想高度一致，但 Partial Rollout 仍标记为 Preview，且项目已经停止新增功能并建议新场景转向 veRL Ascend。
5. **Slime 不作为当前 Ascend 首选。** 它在 NVIDIA/AMD 上的 Agentic RL、Qwen3.6、全异步队列和 SGLang 优化非常强，但没有正式 Ascend 路线，且当前 Fully Async 对中断轨迹仍是重新排队、从头生成，尚未接通真正的断点续推。

### 1.2 推荐的第一版异步语义

第一版 PoC 不应直接启用最激进的“单条轨迹跨多个权重版本继续生成”。推荐从更容易审计的有界异步开始：

- `group_size = 8`
- `optimizer_batch = 16 = 2 × 8`
- 全局队列的最小可训练单元为**完整 prompt group**，不是任意单条轨迹；
- `max_policy_age = 1`
- 轨迹必须保存生成时的 `policy_version` 和逐 token behavior logprob；
- 新请求只使用最新版本；
- 已经在旧版本上开始的轨迹允许在旧版本服务实例上完成；
- 旧版本 group 最多允许进入下一次更新，超过年龄后丢弃或重新生成；
- 初期禁止一条轨迹内部混用两个策略版本；
- 训练启用 rollout correction / Importance Sampling；
- 全局 ready queue 设定上限和反压，避免推理远快于训练造成无限陈旧。

在此基线通过后，再单独评估 Partial Rollout、跨版本续推和更高陈旧度。

---

## 二、问题起点与讨论演进

### 2.1 原始瓶颈：同步 GRPO 被最慢轨迹阻塞

当前同步流程按 prompt 生成一个 GRPO group。若每个 prompt 采样 8 条轨迹，则需要等待这 8 条全部完成，才能：

1. 汇总 8 个 reward；
2. 计算组内均值和标准差；
3. 得到 8 条相对 advantage；
4. 将完整 group 发送给 trainer；
5. 执行反向和参数更新。

当多轮 PI Agent 轨迹包含数据库、知识库、Shell、代码执行或其他工具调用时，不同轨迹的完成时间差异很大。同步 barrier 会让已经空闲的推理卡等待最慢轨迹，训练卡也同时空闲。

讨论最初提出的优化方向是增加 rollout 机器，例如 A、B、C 三个推理节点并行生成。若训练 batch 需要 16 条，期望出现类似：

- A 已完成 7 条；
- B 已完成 6 条；
- C 已完成 5 条；
- 全局累计已经超过 16 条；
- trainer 不再等待某个固定机器或固定小批次全部结束，而是立即开始训练；
- 未完成轨迹继续生成，并进入下一批。

这个方向在系统层面是正确的，但在 GRPO 中必须进一步加上 **group 完整性** 和 **策略版本完整性**，否则会破坏 advantage 计算或引入不可控的 off-policy 偏差。

### 2.2 第二个问题：训练更新时，旧轨迹怎么办

异步化以后，trainer 可能在某些轨迹只完成 10% 时就产生新权重。此时有四种可能：

1. 立即终止旧轨迹并从头重跑；
2. 让旧轨迹使用旧权重完成；
3. 更新推理权重后从中断位置继续，但后续 token 使用新权重；
4. 保留旧轨迹，训练时用 behavior policy logprob 和 Importance Sampling 修正。

讨论中的“已经做到 0.1 进度，不应该浪费”对应 Partial Rollout 或有界旧策略续跑。但它不是纯调度问题，而是算法正确性问题：

- 若单条轨迹的前半段来自策略 `vN`，后半段来自 `vN+1`，则这条轨迹不存在单一 behavior policy；
- 若只保存 trainer 当前策略的 logprob，而不保存真正生成 token 时的 logprob，PPO/GRPO 概率比会错误；
- 若旧轨迹无限滞后，训练数据会逐渐偏离当前策略，可能造成不稳定或失效。

因此，异步系统必须显式维护权重版本、轨迹年龄、behavior logprob、丢弃规则和修正策略。

### 2.3 第三个问题：异步提高的是吞吐，不等于单轨迹变快

增加 rollout 节点和取消同步 barrier 主要改善：

- 集群总吞吐；
- trainer idle ratio；
- rollout 卡利用率；
- 长尾轨迹对 batch 完成时间的影响。

它不会直接缩短一条轨迹的墙钟时间。单轨迹时延还需要单独优化：

- 推理 continuous batching；
- 同一会话稳定路由与 prefix cache；
- 工具调用异步化与连接池；
- 单轮生成上限；
- 总工具调用上限；
- observation 截断；
- Prefill/Decode 分离；
- speculative decoding；
- 受控 finalization；
- 防止单个 assistant turn 吃完整个上下文预算。

本项目已经通过“单回合策略生成上限 4096 + 总轨迹预算 39936 + observation/finalization 保留”处理了一部分单轨迹长尾风险。异步框架不能替代这些轨迹级约束。

---

## 三、当前已验证基线

### 3.1 硬件和软件边界

| 角色 | 机器 | 计算资源 | 已验证拓扑 | 主要软件 |
|---|---|---:|---|---|
| Trainer | 5 号机 | 16 张计算 NPU | `TP8 × PP1 × CP2 × SP` | MindSpeed/Megatron、MS-Swift |
| Rollout | 6 号机 | 16 张计算 NPU | `TP8 × DP2` | vLLM-Ascend 0.23.0 |
| 控制端 | Windows | 不参与训练/推理 | SSH 控制 | 不承载模型、轨迹或 checkpoint |

主训练和 rollout 上下文配置：

- `max_model_len = 40960`
- 总轨迹预算：39936 tokens
- 单回合策略生成上限：4096 tokens
- 策略累计保留：16384
- observation 累计预算：8192
- 单次 observation 上限：2048
- finalization 保留：2048

40K 是当前经过多样样本、checkpoint 恢复和真实在线 GRPO 单步验证的稳定上限。48K 只完成固定 worst1 压力测试；100K 几乎耗尽 64 GiB HBM，不具有工程安全余量。

### 3.2 同步权重发布协议

当前权重同步已经具备异步系统需要的大部分基础设施：

1. Trainer 输出扁平 LoRA 和 manifest；
2. 记录 tensor 数、字节数、SHA256 和唯一 `transfer_id`；
3. 5 号机通过服务器内网直接 rsync 到 6 号 `.incoming`；
4. 6 号机复算 SHA256 和字节数；
5. 通过后发布到不可变版本目录；
6. 原子替换 `adapter_flattened.pt` symlink；
7. rollout worker 全部确认加载成功；
8. 6 号向 5 号返回版本化 ACK；
9. Trainer 收到 `all_workers_loaded=true` 后才允许生成轨迹。

这套协议适合作为异步系统 `ParameterSynchronizer` 的原型，但需要扩展：

- 显式 `policy_version`；
- 多版本并存；
- 新旧 rollout worker 的路由；
- 旧版本 drain；
- 队列中轨迹与版本的绑定；
- checkpoint 时保存队列和版本状态。

### 3.3 最新多提示 40K 单步证据

最新修正版使用两个真实 prompt，每组 8 条：

| 指标 | 结果 |
|---|---:|
| 总轨迹数 | 16 |
| group 数 | 2 |
| reward 分布 | `0×4 / 0.2×3 / 1×9` |
| 总体 reward mean | 0.6000 |
| 总体 reward std | 0.3618 |
| advantage | 16/16 非零且 finite |
| loss | `1.1174e-4` |
| KL | `7.754e-5` |
| grad norm | `0.06853` |
| HF LoRA-B | 204 个 tensor 中 176 个非零 |
| DCP BF16/FP32 fused B | 108 个中 80 个非零 |
| HF adapter SHA256 | `eddb8468ac5874eeef7931adf5bb803573867bd1ecab04faedabb5449ff64cf8` |

该结果证明：

- 现有同步系统具有有效 reward 方差；
- GRPO group 语义正确；
- advantage 有效；
- trainer 真实更新参数；
- 5→6 权重发布可用；
- 40K 多轮工具轨迹链路可用。

它没有证明：

- 长跑后 heldout 质量提升；
- 多 rollout 机器全局队列可用；
- 旧策略轨迹可安全复用；
- trainer 与 rollout 可以持续重叠；
- 当前 MS-Swift 代码已经具备全异步能力。

---

## 四、目标异步架构

### 4.1 组件

目标系统至少应包含六个逻辑组件：

| 组件 | 职责 |
|---|---|
| Prompt/Group Scheduler | 为每个 prompt 创建 `group_id`，分发 8 个 sibling rollout |
| Rollout Router | 将请求路由到 A/B/C 等推理节点，维护会话亲和与版本路由 |
| Global Group Buffer | 接收单条轨迹，按 `group_id` 聚合，只有完整 group 才进入 ready queue |
| Reward/Verifier Pool | 异步计算 reward，记录 verifier 版本和失败原因 |
| Trainer | 每次取两个完整 group，执行 GRPO/PPO 更新 |
| Parameter Synchronizer | 发布新 LoRA 版本、管理旧版本 drain、ACK 和回滚 |

逻辑数据流：

```text
Prompt Dataset
      |
      v
Group Scheduler -- group_id / sample_id / target_policy_version
      |
      +----------+----------+
      |          |          |
 Rollout A   Rollout B   Rollout C
      |          |          |
      +----------+----------+
                 |
                 v
        Reward / Verifier Pool
                 |
                 v
        Global Group Buffer
        - PARTIAL groups
        - READY groups
        - STALE groups
                 |
          2 complete groups
                 |
                 v
              Trainer
                 |
          policy vN -> vN+1
                 |
                 v
        Parameter Synchronizer
```

### 4.2 最小可训练单元必须是完整 group

设：

- 每个 prompt 的生成数为 `G = 8`；
- trainer 每步消费 `B = 16` 条轨迹；
- 则每步必须消费 `K = B / G = 2` 个完整 prompt group。

GRPO 组内 advantage 可写为：

```text
A(i,j) = [r(i,j) - mean(r(i,1..G))] / [std(r(i,1..G)) + epsilon]
```

其中 `i` 是 prompt group，`j` 是同一 prompt 的第 `j` 条生成。

所以 A、B、C 三个 rollout 节点的“7、6、5 条”不能直接解释为任意 18 条即可训练。正确条件是：

```text
ready_group_count >= 2
```

而不是：

```text
completed_trajectory_count >= 16
```

同一 group 的 8 个成员可以分散到不同机器，但必须具有一致的：

- `group_id`
- `prompt_id`
- reward contract
- sampling contract
- tokenizer/chat template
- target policy/version 约束

### 4.3 推荐的轨迹状态机

```text
PENDING
  -> RUNNING(policy=vN)
  -> PARTIAL(policy_segments=[vN, ...])       # 后续阶段才允许
  -> GENERATED
  -> REWARDED
  -> GROUP_READY
  -> LEASED_BY_TRAINER
  -> CONSUMED
```

异常分支：

```text
RUNNING/GENERATED
  -> RETRYABLE_FAILED
  -> REQUEUED

RUNNING/GENERATED/REWARDED
  -> STALE
  -> EXPIRED 或 REGENERATE

LEASED_BY_TRAINER
  -> LEASE_TIMEOUT
  -> GROUP_READY
```

所有状态迁移都必须幂等。建议唯一键至少包含：

```text
(run_id, group_id, sample_id, attempt_id, policy_version)
```

### 4.4 必须保存的数据字段

每条轨迹至少保存：

| 类别 | 字段 |
|---|---|
| 身份 | `run_id`、`prompt_id`、`group_id`、`sample_id`、`attempt_id` |
| 策略 | `policy_version`、`base_hash`、`adapter_hash`、`weight_loaded_at` |
| 生成 | token IDs、逐 token rollout logprob、sampling 参数、finish reason |
| 多轮 | turn 边界、tool call、observation 边界、loss mask |
| 奖励 | reward、reward components、verifier version、reward timestamp |
| 调度 | submit/start/finish 时间、rollout node、retry count、queue age |
| 训练 | consumed step、trainer policy version、IS ratio 统计、drop reason |

禁止只保存最终文本。对于多轮工具轨迹，必须保留实际由推理引擎生成的 token IDs，避免 text→token 二次编码造成模板和特殊 token 不一致。

---

## 五、权重版本与 off-policy 正确性

### 5.1 为什么异步 GRPO 需要 behavior logprob

同步 GRPO 假设轨迹来自当前旧策略 `π_old`。异步时，轨迹可能实际来自更早的 `π_behavior`。

若仍使用错误的分母，概率比：

```text
ratio = π_train(a|s) / π_old(a|s)
```

就不再对应真正生成该 action 的分布。异步训练应基于真正的 behavior logprob，并按框架支持使用截断 Importance Sampling、token mask 或 Decoupled PPO 等修正。

第一版 PoC 必须满足：

- rollout 端直接返回逐 token logprob；
- 不允许 trainer 事后用当前权重伪造 old logprob；
- 记录每条样本的策略版本年龄；
- 监控 IS ratio 分布和被 clip/mask 的 token 比例。

### 5.2 三种可选的权重更新策略

#### 策略 A：Drain-and-Swap

- 停止提交新请求；
- 等旧版本所有 in-flight 轨迹完成；
- 更新全部 rollout worker；
- 恢复新请求。

优点：语义最简单、接近 on-policy。  
缺点：重新引入长尾 barrier，异步收益有限。

#### 策略 B：有界旧版本续跑，推荐作为第一版

- 新权重 `vN+1` 发布后，新请求进入 `vN+1` worker；
- 已在 `vN` 上开始的请求继续由 `vN` worker 完成；
- `vN` group 最多允许用于 `vN+1` 的一次训练；
- 超过 `max_policy_age=1` 后丢弃或重跑；
- trainer 使用真正的 rollout logprob 和 IS correction。

优点：避免浪费已进行的轨迹，又不让一条轨迹混用多个策略版本。  
缺点：需要短期同时保留两个 LoRA 版本和明确的版本路由。

#### 策略 C：跨版本 Partial Rollout

- 轨迹在 `vN` 上生成一部分；
- 权重更新后从 KV/中间状态继续；
- 后续 token 使用 `vN+1`；
- 每个 token 或 segment 保存实际 behavior policy 信息；
- 使用 Decoupled PPO 或等价修正。

优点：最少等待、最大重叠。  
缺点：实现和审计最复杂，容易出现混合版本、KV 一致性和 logprob 错配。

本报告不建议第一版直接采用策略 C。

### 5.3 三节点滚动更新建议

若未来有 A/B/C 三个 rollout 节点，可以使用蓝绿/滚动更新：

1. A 停止接收 `vN` 新请求，完成已有请求；
2. A 加载 `vN+1` 并开始接收新请求；
3. B、C 继续完成 `vN` 请求；
4. 依次更新 B、C；
5. 全局 router 按 `policy_version` 和 session affinity 路由；
6. `vN` 没有 in-flight 后卸载；
7. 队列中超过年龄的 `vN` group 失效。

这种方式比同时停止三台更新更适合 Agent 长尾，但要求路由器和队列都认识策略版本。

---

## 六、框架调研

### 6.1 结论矩阵

| 框架 | 全局流式队列 | 训推重叠 | Partial Rollout | 陈旧度/修正 | 多轮 Agent | Ascend 适配 | 对本项目判断 |
|---|---|---|---|---|---|---|---|
| veRL | 强 | 强 | 强 | 强 | 强 | 强 | 第一 PoC 候选 |
| AReaL | 强 | 原生全异步 | 强 | 强 | 很强 | 有正式分支 | 第二候选，架构最完整 |
| ROLL | 强 | 强 | 中 | 中到强 | 强 | 强 | 第三候选 |
| MindSpeed-RL | 中到强 | 中 | 有但 Preview | 中 | 有 | 原生 | 只作参考 |
| Slime | 强 | 强 | 当前全异步路径未接通续推 | 中到强 | 很强 | 无 | NVIDIA 场景优先，当前不选 |
| NeMo RL | 强 | 强 | 强 | 很强 | 强 | 无 | 算法和缓冲区参考 |
| OpenRLHF | 强 | 强 | 强 | 强 | 强 | 无正式路线 | 当前不选 |

“强/中”是基于截至 2026-07-29 的公开文档和代码能力判断，不是本项目在同一硬件、同一模型上的实测排名。

### 6.2 veRL

veRL Fully Async 由四部分组成：

- Rollouter
- MessageQueue
- Trainer
- ParameterSynchronizer

关键行为与本项目目标高度一致：

- Rollouter 按样本流式生产；
- Trainer 从 MessageQueue 持续消费；
- 凑够 `require_batches × ppo_mini_batch_size` 后训练；
- rollout 和 trainer 同时运行；
- `staleness_threshold` 控制旧样本比例；
- `partial_rollout` 可在权重同步时中断请求，更新后继续；
- rollout 端保存 old logprob；
- 可启用 rollout Importance Sampling；
- 有 trainer/rollouter idle ratio、stale sample 和 partial span 指标。

Ascend 文档已经给出四种常用组合：

- FSDP2 + vLLM-Ascend
- Megatron + vLLM-Ascend
- FSDP2 + SGLang Ascend
- Megatron + SGLang Ascend

同时，Agentic RL 支持 server-based async rollout、多轮工具调用和 token-based API。

**适配优势：**

- 当前项目已经使用 Megatron/MindSpeed、vLLM-Ascend、Ray/HTTP 异步思想和跨机权重同步；
- veRL 的 Ascend 路径与现有技术栈重合度最高；
- 可先复用现有 reward contract 和 PI AgentLoop，再替换调度与 trainer orchestration。

**待验证风险：**

- Fully Async 与 Ascend 的交叉组合是否有完整端到端 CI；
- Qwen3.6-27B 的模型结构和 GDN/LoRA target；
- 40K、多轮工具、逐 token logprob；
- 跨机 LoRA 动态 refit；
- 当前 vLLM-Ascend 0.23.0 与 veRL 推荐版本是否一致；
- `TP8 × CP2` trainer 与 `TP8 × DP2` rollout 的权重重切分/同步。

### 6.3 AReaL

AReaL 的设计目标就是大规模全异步强化学习。其关键机制包括：

- 推理与训练在不同设备上持续并行；
- `max_head_offpolicyness` 限制 rollout 最多落后多少权重版本；
- 支持 partial rollout，即一条轨迹可以分段跨策略版本；
- Decoupled PPO 处理策略版本错位；
- 支持 GRPO、PPO、DAPO、GSPO 等同步/异步模式；
- OpenAI 兼容代理自动捕获 token、logprob、会话树和 reward；
- 多轮 Agent 可以按 completion 分配奖励，也可以对早期轮次折扣。

项目在 2026 年宣布维护 Ascend 分支。

**适配优势：**

- 对“多轮 Agent + 旧策略 + partial trajectory”的理论和系统支持最完整；
- 更接近长期目标，而不是只解决单次 batch pipeline；
- Agent 代理接口有利于复用现有 PI 工作流。

**待验证风险：**

- Ascend 支持位于独立分支，与主线 2.0 微服务架构的同步程度；
- Qwen3.6 与当前 LoRA 权重格式；
- Megatron/MindSpeed checkpoint 兼容；
- 现有 vLLM-Ascend 版本；
- 40K Agent proxy 的 token 一致性和性能。

### 6.4 阿里 ROLL

ROLL 的 Agentic 异步并行以 EnvManager 为单位：

- 每个环境独立执行 rollout loop；
- 环境之间没有同步 barrier；
- `rollout_scheduler.get_batch()` 等到需要的轨迹数；
- 异步模式下，即使一个 batch 已经返回给训练，EnvManager 仍继续执行；
- `async_generation_ratio` 控制提前生成多少批；
- 支持 GRPO 和多种 off-policy variant；
- 已提供 Ascend A2/A3/950 的安装和 vLLM-Ascend 组合。

**适配优势：**

- 大厂维护、Agentic pipeline 完整；
- Ascend 路线明确；
- 全局 scheduler 的行为与“多个 rollout 节点凑 batch”直接对应。

**待验证风险：**

- 公开资料中对 partial trajectory、token 级混合策略版本和 checkpoint queue state 的描述较少；
- `async_generation_ratio` 更像批级 ahead generation，需要确认是否满足本项目希望的样本/group 级流式消费；
- Qwen3.6 和 40K 组合仍需实测。

### 6.5 MindSpeed-RL

MindSpeed-RL 的 Partial Rollout 文档几乎直接描述了本项目最初提出的方案：

- 长序列提前中断；
- 截断样本进入 TransferDock；
- 当完成 prompt 数达到 GBS 时进入训练；
- 未完成样本在下一轮优先续推；
- 异步引擎可按样本粒度返回；
- 混合旧的截断样本和新样本重新调度。

它还支持 ReTool、Search Tool、多轮工具调用和异步引擎。

但需要注意：

- Partial Rollout 标记为 Preview；
- 多项数据调度和重切分能力也标记为 Preview；
- 2026 年 4 月项目宣布停止新增功能集成；
- 官方建议体验新的 Ascend RL 方案时使用 veRL Ascend。

因此不建议把新系统绑定到已停止扩展的主干，但可以参考它的 TransferDock、截断调度和 Ascend 实现。

### 6.6 Slime

Slime 的 Fully Async worker：

- 在进程内维护持续运行的 asyncio worker；
- 保持固定数量的 in-flight generations；
- 完成的完整 group 进入输出队列；
- 每次训练调用从队列取够 `rollout_batch_size`；
- 最慢轨迹不会阻塞下一训练步；
- 支持自定义多轮 Agent、工具、sandbox、test-based reward；
- 支持 Qwen3.6；
- 支持多 Agent、SGLang、PD 分离和 session-affinity。

但其公开限制明确写明：

- `ABORTED` 轨迹的 partial-rollout resume 尚未接通；
- 当前会重新入队并从头开始；
- 官方硬件支持以 NVIDIA B/H 系列和部分 AMD MI300/MI325 为主；
- 没有正式 Ascend 后端。

结论是：如果未来改用 NVIDIA 集群，Slime 会是非常强的候选；在当前 Ascend 环境中，为 Slime 移植 Megatron、SGLang、权重同步和 NPU kernel 的成本过高。

### 6.7 NeMo RL 与 OpenRLHF

NeMo RL 的 Async GRPO 提供很好的参考实现：

- replay buffer 按 prompt group 存储；
- 每个条目就是 `num_generations_per_prompt` 个 sibling；
- 记录 generation weight version 和 target weight version；
- `max_trajectory_age_steps` 控制最大年龄；
- 样本不足时 trainer 等待；
- 支持 in-flight weight updates；
- 官方要求异步时启用 Importance Sampling correction。

OpenRLHF 支持：

- async queue；
- multi-turn Agent；
- partial rollout；
- vLLM pause/resume；
- rollout IS correction；
- 允许 in-flight 样本包含新旧权重 token。

两者都缺少适合当前项目的正式 Ascend 路线，因此只适合作为算法、队列和指标设计参考。

---

## 七、为什么当前 MS-Swift 链路不能直接满足目标

当前实现中的异步主要集中在：

- Python `asyncio` 并发请求；
- 多个 vLLM endpoint；
- 多轮 Agent 调用与工具执行；
- 跨机 LoRA 同步；
- trainer 等待完整 generation batch。

它还缺少全异步 RL 系统必须具备的核心抽象：

- 跨 rollout 节点的持久化全局 Group Buffer；
- trajectory/group 的显式状态机；
- behavior policy version；
- replay buffer 与 maximum age；
- rollout logprob 的完整回传和校验；
- Importance Sampling / Decoupled PPO；
- trainer 与 rollout 的持续并行控制器；
- 多版本 rollout worker；
- queue/checkpoint 一致性恢复；
- stale group 丢弃和重生成；
- 全局反压。

因此，继续在当前 scheduler 上小修，可以实现“更多并发 endpoint”，但难以安全地演进成可恢复、可审计的全异步训练系统。更合理的方式是：

1. 保留当前同步实现作为质量和工程基线；
2. 在专业异步 RL 框架上做 PoC；
3. 复用本项目已经验证的 PI Agent、reward contract、loss mask、40K 预算和权重发布协议；
4. 不自研一整套 trainer/rollout/replay/off-policy 基础设施，除非三个候选框架都无法通过硬门槛。

---

## 八、推荐的 PoC 设计

### 8.1 PoC 目标

PoC 只回答四个问题：

1. 多个 rollout worker 是否可以向同一个 Group Buffer 供给 sibling trajectories；
2. 凑够两个完整 group 后，trainer 是否可以在 rollout 继续运行时更新；
3. 权重更新后，新旧轨迹是否按版本和年龄规则被正确消费或丢弃；
4. 与当前同步基线相比，是否降低 trainer idle ratio，同时不破坏 reward、advantage 和参数更新。

PoC 不回答最终模型质量提升；质量提升仍需长跑和独立 heldout。

### 8.2 第一阶段：veRL-NPU 单步兼容性

建议最小配置：

| 参数 | 建议值 |
|---|---|
| 模型 | Qwen3.6-27B |
| 上下文 | 先 8K 兼容 smoke，再 40K |
| group size | 8 |
| optimizer batch | 16 |
| prompt 数 | 2 个已验证 prompt 2/10 |
| trainer | 5 号机 16 NPU |
| rollout | 6 号机 `TP8 × DP2` |
| policy age | 先 0，再 1 |
| partial rollout | 第一轮关闭 |
| rollout logprob | 必须开启 |
| IS correction | 必须开启 |
| warmup | 1-step smoke 显式为 0 |

验收：

- Qwen3.6 模型和 tokenizer/template 正常；
- PI AgentLoop 能完成多轮工具轨迹；
- tool observation 不进入 loss；
- assistant content/tool call 正确进入 loss mask；
- 2 个完整 group；
- reward 方差非零；
- advantage 16/16 finite；
- trainer/rollouter 实际重叠；
- 参数真实变化；
- weight version 和 adapter hash 可追溯；
- 失败后可停止且无 NPU 残留。

### 8.3 第二阶段：有界异步

开启：

- `max_policy_age = 1`
- queue depth 上限建议从 2～4 个 optimizer batch 开始；
- 新请求使用最新版本；
- 旧请求在旧版本 worker 上完成；
- 记录 stale sample/group；
- 超龄 group 丢弃并统计；
- checkpoint 保存 trainer step、当前/旧策略版本、ready/partial queue 索引。

验收：

- 至少出现一组 `vN` 轨迹被 `vN+1` trainer 合法消费；
- IS ratio finite；
- clip/mask 比例在可解释范围；
- 无跨 group 混合；
- 无同一样本重复消费；
- 重启后不丢失或重复训练 ready group。

### 8.4 第三阶段：Partial Rollout

只有第二阶段稳定后才开启。重点验证：

- 中断位置和恢复位置的 token 精确一致；
- KV cache 与新权重更新方式明确；
- 每个 token/segment 的 policy version 可追溯；
- behavior logprob 与生成 token 一一对应；
- mixed-version trajectory 的 loss 修正经过单元测试；
- 与“不跨版本、允许旧请求完成”的对照相比，吞吐收益足够大。

若收益很小或修正不稳定，应保留第二阶段方案，不追求最激进异步。

### 8.5 AReaL 备用验证

若 veRL 在以下任一硬门槛失败，则启动 AReaL Ascend PoC：

- Qwen3.6 模型结构不兼容；
- Ascend Fully Async 无法运行；
- 40K AgentLoop 无法返回正确 token/logprob；
- LoRA refit 或跨机权重同步无法接入；
- group queue 不能满足当前 reward contract；
- trainer 与 rollout 无法真正重叠。

AReaL PoC 应优先验证 `max_head_offpolicyness=1`，暂不直接使用典型范围 2～8。

---

## 九、验收指标与监控

### 9.1 正确性指标

| 指标 | 门槛 |
|---|---|
| group 完整率 | Trainer 消费的 group 必须 8/8 |
| group 混淆 | 0 |
| 重复消费 | 0 |
| 丢失 ready group | 0 |
| reward/advantage finite | 100% |
| LoRA-B 真实变化 | 必须 |
| rollout token/logprob 对齐 | 100% |
| adapter hash/版本可追溯 | 100% |
| 超龄样本消费 | 0 |
| 失败后 NPU 残留 | 0 |

### 9.2 性能指标

- rollout trajectories/min
- completed groups/min
- group ready latency p50/p95/p99
- trainer idle ratio
- rollouter idle ratio
- ready queue depth
- partial queue depth
- policy age 分布
- stale/drop/retry rate
- weight sync wall time
- tool latency p50/p95/p99
- 单轨迹 action/observation tokens
- 每 step 总墙钟时间
- NPU 利用率和 HBM 峰值

### 9.3 算法稳定性指标

- reward mean/std，按 prompt group 报告；
- `frac_reward_zero_std`；
- advantage mean/std/min/max；
- KL；
- rollout policy 与 trainer policy 的版本差；
- IS ratio 分布；
- IS clip rate/token mask rate；
- old-policy group 对总 batch 的占比；
- response length 与截断率；
- tool success、required-table coverage、gold evidence hit；
- full success、exact result、process score；
- 按 DWH/KB/Hybrid 分层。

---

## 十、风险与限制

### 10.1 当前结论不是同硬件基准测试

框架排名来自官方文档、代码结构和硬件支持矩阵，不是使用 Qwen3.6-27B、相同 Ascend NPU、相同 40K Agent 任务的实测性能比较。因此不能把“第一候选”理解为已证明最快。

### 10.2 “支持 Ascend”和“支持 Fully Async”不自动等于交叉组合已验证

veRL、AReaL、ROLL 分别具备 Ascend 和异步能力，但仍需验证：

- Qwen3.6；
- 40K；
- Megatron/MindSpeed；
- vLLM-Ascend 当前版本；
- LoRA 动态 refit；
- 多轮工具；
- rollout logprob；
- 全异步；
- 两机或三机拓扑。

### 10.3 异步可能降低样本新鲜度

推理资源增加过多，而 trainer 吞吐不变时，队列会堆积，旧数据比例上升。更多 rollout 机器不一定带来更好训练，必须通过反压和 maximum age 限制。

### 10.4 Partial Rollout 可能牺牲可解释性

跨策略版本续推虽然减少等待，但会增加：

- token 级策略归属复杂度；
- KV cache 正确性风险；
- IS 方差；
- checkpoint 恢复难度；
- bug 定位成本。

第一版不采用它，是风险控制而不是否定其价值。

### 10.5 工程吞吐不能代替模型质量

即使异步后每小时生成更多轨迹，也不能单凭：

- reward 上升；
- training loss；
- KL；
- throughput；
- NPU 利用率；

证明模型质量提升。正式结论仍需 matched continued-SFT、普通 DPO、verified RPO、trajectory-GRPO 和独立 heldout 对照。

---

## 十一、决策与执行建议

### 11.1 当前正式决策

1. 保留现有同步 MS-Swift/MindSpeed GRPO，作为工程和质量基线。
2. 不在当前 scheduler 上直接堆叠复杂的全异步 replay/off-policy 逻辑。
3. 首先实施 veRL-NPU 最小 PoC。
4. AReaL Ascend 作为第二候选和算法参考。
5. ROLL 作为第三候选。
6. MindSpeed-RL 只参考 TransferDock 和 Partial Rollout 设计。
7. 当前 Ascend 环境不采用 Slime；若未来转 NVIDIA，再重新评估。
8. 不在完成单步、有界异步、恢复和 heldout 门禁前启动长跑。

### 11.2 推荐实施顺序

```text
同步基线冻结
   -> veRL 8K 单步兼容
   -> veRL 40K 同步 Agent GRPO
   -> veRL 全局 group queue，policy_age=0
   -> veRL 有界异步，policy_age=1
   -> 故障恢复与 queue checkpoint
   -> 与当前同步基线做同任务吞吐/正确性对比
   -> 决定是否验证 Partial Rollout
   -> 小规模质量 pilot
   -> 独立 heldout
   -> 才决定正式长跑
```

---

## 十二、待回答问题

在进入实现前，需要通过代码或 PoC 回答：

1. veRL Ascend 的 Fully Async 是否已经覆盖 Megatron + vLLM-Ascend？
2. Qwen3.6 的 GDN 层、LoRA `linear_qkv + linear_fc1` 能否直接适配？
3. rollout backend 能否返回 PI 多轮工具轨迹的逐 token logprob？
4. 40K 下 AgentLoop 的 token、template 和 loss mask 是否与当前 scheduler 一致？
5. LoRA refit 是否支持两个 `TP8` rollout 副本，并能返回全 worker ACK？
6. veRL 默认的 `ppo_mini_batch_size` 语义是轨迹数还是 group 数；如何保证 8 个 sibling 不被拆错？
7. queue checkpoint 是否包含未完成 group、ready group 和 policy version？
8. 旧版本 rollout worker 是否能与新版本短期并存？
9. Ascend 下 Partial Rollout 的 sleep/resume 和 KV cache 更新是否可靠？
10. 三台 rollout 节点加入后，瓶颈会转移到工具服务、reward verifier 还是 trainer？

---

## 十三、证据与参考资料

### 13.1 本项目证据

- [项目 README：在线 GRPO、40K、多机权重同步和真实参数更新记录](../README.md)
- [Qwen3.6-27B DPO 长上下文验证结果](../长上下文实验报告/Qwen3.6-27B_DPO长上下文验证结果_20260723.md)

### 13.2 框架官方资料

- [veRL Fully Async](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md)
- [veRL Agentic RL](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst)
- [veRL Ascend Quick Start](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/get_start/quick_start.rst)
- [AReaL 异步强化学习](https://areal-project.github.io/AReaL/zh/algorithms/async.html)
- [AReaL Agent Workflow](https://areal-ai.io/AReaL/zh/reference/agent_workflow.html)
- [AReaL 官方仓库与 Ascend 公告](https://github.com/areal-project/AReaL)
- [ROLL Agentic Asynchronous Parallel Rollout](https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/async_parallel_rollout/)
- [ROLL Asynchronous Training](https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/async_training/)
- [ROLL Ascend](https://alibaba.github.io/ROLL/docs/User%20Guides/Hardware%20Support/ascend_usage/)
- [MindSpeed-RL Partial Rollout](https://github.com/Ascend/MindSpeed-RL/blob/master/docs/zh/features/partial_rollout.md)
- [MindSpeed-RL 多轮工具](https://github.com/Ascend/MindSpeed-RL/blob/master/docs/zh/features/multi_turn.md)
- [Slime Fully Async](https://thudm.github.io/slime/_examples_synced/fully_async/README.html)
- [Slime Agentic RL](https://thudm.github.io/slime/get_started/agent.html)
- [Slime Hardware Support](https://thudm.github.io/slime/get_started/quick_start.html)
- [NeMo RL Async GRPO](https://docs.nvidia.com/nemo/rl/latest/guides/async-grpo.html)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)

---

## 十四、报告边界

本报告对以下事项做出结论：

- 目标异步架构是否合理；
- GRPO group 与策略版本的正确处理方式；
- 当前框架候选和推荐顺序；
- 第一版 PoC 的范围、指标和门禁。

本报告没有授权：

- 修改正式训练框架；
- 占用新的服务器或 NPU；
- 启动异步长跑；
- 宣称异步训练会提高模型质量；
- 宣称某个框架已经在本项目环境中验证通过。

下一项可执行工作应是 **veRL-NPU 兼容性审计与最小单步 PoC 设计**，并继续遵守服务器内数据面、`llin-*` 资源隔离、hash/manifest、原子发布、停止清理和独立 heldout 规则。
