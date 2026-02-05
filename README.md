# 项目流程说明（SkeletonSteering）

本文件描述从数据与骨骼模型到 steering 任务训练、模型保存与评估的完整流程，并标注关键命令。

提示：本文件包含 Mermaid 代码块；在支持 Mermaid 的 Markdown 查看器（如 GitHub、VS Code + Mermaid 插件）中可直接渲染。若无法渲染，可参考后面的“纯文本备份”。

## 总览流程（Mermaid 流程框图）

```mermaid
flowchart TD
  %% =========================
  %% Offline dataset pipeline
  %% =========================
  subgraph Offline["离线数据准备（可选）"]
    KIT["KIT 原始动作数据<br/>（.npz clips）"]
    Extract["scripts/extract_kit_expert_dataset.py<br/>筛选/复制动作片段"]
    Dataset["expert_dataset/<br/>本地子集"]
    Retarget["scripts/retarget_kit_expert_dataset.py<br/>retarget + 拼接 Trajectory"]
    KitTraj["training/kit_expert_traj.npz<br/>（custom_traj_path）"]

    KIT --> Extract --> Dataset --> Retarget --> KitTraj
  end

  %% =========================
  %% Training
  %% =========================
  subgraph Train["训练（AMP + PPO · JAX）"]
    Conf["training/amp_ppo_jax_conf.yaml<br/>（可用 --env_id 覆盖）"]
    EntryTrain["training/train_amp_ppo_jax.py"]
    TrainFn["training/amp_ppo_jax.py: train()"]
    BuildEnv["TaskFactory.make()<br/>+ LogWrapper + VecEnv (MJX)"]

    ExpertTraj["专家轨迹<br/>env.unwrapped().th.traj"]
    CacheTraj["results_dir/expert_traj.npz<br/>（缓存）"]

    DiscSpec["_build_disc_obs_spec()<br/>key_bodies / key_sites"]
    ExpertDisc["_build_expert_disc_obs()<br/>预计算判别器观测"]
    CacheDisc["results_dir/expert_disc_obs.npz<br/>（缓存）"]

    Loop["训练迭代循环"]
    Rollout["_rollout()<br/>并行采样 trajectories"]
    DiscR["_compute_disc_reward()"]
    TotalR["组合奖励<br/>task_weight·task + disc_weight·disc"]
    GAE["_compute_gae()<br/>adv / returns"]
    DiscBatch["构造 disc_batch<br/>demo + agent + replay"]
    Update["_update_params()<br/>PPO + Disc + optax 更新"]
    Log["stdout / TensorBoard"]
    Ckpt["保存 checkpoint<br/>amp_ppo_jax_best.pkl / interval"]

    Conf --> EntryTrain --> TrainFn --> BuildEnv
    KitTraj -. "custom_traj_path" .-> BuildEnv

    BuildEnv --> ExpertTraj --> CacheTraj
    ExpertTraj --> DiscSpec --> ExpertDisc --> CacheDisc

    CacheDisc --> Loop
    Loop --> Rollout --> DiscR --> TotalR --> GAE --> DiscBatch --> Update --> Log --> Ckpt
    Log --> Loop
  end

  %% =========================
  %% Evaluation / Recording
  %% =========================
  subgraph Eval["评估 / 录制"]
    Best["training/logs/<run>/amp_ppo_jax_best.pkl"]
    EntryEval["training/eval_amp_ppo_jax.py"]
    Load["load_checkpoint()"]
    MakeEnv["_make_env()"]
    Step["循环 step<br/>actor 输出动作 → env.step"]
    Record{"--record ?"}
    Video["保存视频到<br/>training/mushroom_rl_recordings/"]

    Best --> EntryEval --> Load --> MakeEnv --> Step --> Record
    Record -- Yes --> Video
    Record -- No --> Step
  end
```

## 总览流程（纯文本备份）

```
[骨骼模型]
    |
    v
[KIT 数据集] --> [重定向/插值/关节筛选] --> [专家轨迹缓存]
    |                                          |
    |                                          v
    |                                  expert_traj.npz
    |                                  expert_disc_obs.npz
    v
[steering 任务]  (SkeletonTorque.walk.mocap)   ★重点
    |
    v
[算法：AMP + PPO (JAX)]                        ★重点
    |
    v
[训练脚本] train_amp_ppo_jax.py                ★重点
    |
    v
[模型保存 + 日志] training/logs/<run>/
    |
    v
[评估] eval_amp_ppo_jax.py
```

## 训练内部循环（Mermaid）

```mermaid
flowchart TD
  S["TrainState<br/>params / opt_state / normalizers / replay / rng"] --> R["rollout_fn = _rollout()<br/>采样 (obs, action, logp, value, reward, done, disc_obs)"]
  R --> DR["disc_reward = _compute_disc_reward(disc_obs)"]
  DR --> TR["total_reward = task_weight·task + disc_weight·disc"]
  TR --> G["GAE: _compute_gae(total_reward, values, dones, last_value)<br/>得到 adv / returns"]
  G --> B["PPO batch<br/>obs, actions, logp, adv, returns"]
  G --> DB["disc_batch<br/>agent samples + replay samples + demo samples"]
  DB --> N{"sample_count < normalizer.samples ?"}
  N -- Yes --> UN["更新 obs_norm / disc_norm"]
  N -- No --> RB["更新 replay buffer"]
  UN --> RB
  B --> U["update_fn = _update_params()<br/>actor/critic/disc 更新"]
  DB --> U
  RB --> U
  U --> M["metrics<br/>actor_loss / value_loss / disc_loss / KL / clip_frac ..."]
  M --> L["log + checkpoint"]
  L --> S
```

## 关键组件与目录

- 骨骼模型
  - 基于 LocoMuJoCo 的 SkeletonTorque 相关模型与环境定义。
  - 主要运行依赖来自 `loco_mujoco`（conda env: `skeleton1.0.1`）。

- KIT 数据集
  - 原始/处理后的 KIT 运动数据位于 `expert_dataset/`。
  - 训练时会加载数据并进行重定向、插值与关节筛选。

- 专家轨迹缓存
  - `training/kit_expert_traj.npz`：KIT 轨迹的本地缓存（用于加速加载）。
  - 运行训练时会在 `training/logs/<run>/` 生成：
    - `expert_traj.npz`：当前环境加载并处理后的专家轨迹缓存。
    - `expert_disc_obs.npz`：按判别器观测配置预计算的特征缓存。

- Steering 任务（重点）
  - 任务定义位于 `training/steering_task.py`。
  - 训练环境 ID 典型为：`SkeletonTorque.walk.mocap`。
  - 任务目标：在指定速度区间内，跟随目标方向（steering）。

- 算法（重点）
  - `training/amp_ppo_jax.py`：JAX 版本 AMP + PPO 实现。
  - AMP：使用判别器对专家轨迹与策略轨迹进行区分。
  - PPO：作为策略优化主体，结合 AMP 奖励信号。

- 训练（重点）
  - 训练脚本：`training/train_amp_ppo_jax.py`
  - 配置文件：`training/amp_ppo_jax_conf.yaml`

## 训练命令（关键）

```
python training/train_amp_ppo_jax.py \
  --config training/amp_ppo_jax_conf.yaml \
  --env_id SkeletonTorque.walk.mocap \
  --seed 42
```

可复现性更强（但可能更慢）的配置示例：

```
# GPU + 尽量确定
python training/train_amp_ppo_jax.py \
  --config training/amp_ppo_jax_conf.yaml \
  --deterministic --seed 42

# CPU（更慢但更可复现）
python training/train_amp_ppo_jax.py \
  --config training/amp_ppo_jax_conf.yaml \
  --deterministic --jax_cpu --seed 42
```

## 模型保存与日志

- 训练输出位于：`training/logs/<run>/`
- 常见产物：
  - `amp_ppo_jax_best.pkl`：最佳模型权重。
  - `tensorboard/`：TensorBoard 事件文件。
  - `expert_traj.npz` / `expert_disc_obs.npz`：专家轨迹缓存。

## 评估命令（关键）

```
python training/eval_amp_ppo_jax.py \
  --model_path training/logs/<run>/amp_ppo_jax_best.pkl \
  --n_steps 2000 --deterministic
```

录制视频：

```
MUJOCO_GL=glfw python training/eval_amp_ppo_jax.py --record \
  --model_path training/logs/<run>/amp_ppo_jax_best.pkl
```

## 监控与可视化

- TensorBoard：

```
tensorboard --logdir training/logs
```

## 备注

- 若出现 NaN，通常来自环境数值爆炸或优化溢出；可开启 JAX NaN 调试并降低学习率/裁剪动作。
- steering 任务、AMP + PPO 算法与训练脚本是整个流程的核心路径。
