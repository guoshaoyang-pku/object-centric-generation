# src

两层代码：

- `src/*.py`：2026-10-27 起的训练、稳定化、诊断、oracle 与数据扩充代码，从原项目的 `research/20261027_claude/` 原样复制。集群控制脚本（supervisor、配额处理）不在这里。
- `src/code_base_v4/`：上面这些脚本导入的 codec、渲染器、小 flow 和评测模块，从不可变快照 `research/20261027/snapshots/code_v4/` 原样复制。副本与快照 `checksums.json` 中的 SHA256 逐个一致。

脚本默认在自己旁边找 `code_base_v4/`，也可以用环境变量 `PCV_CODE_BASE` 指向别处。`SHA256SUMS` 记录了本目录全部 `.py` 在复制时的哈希。

## 实验脚本（src/）

| 文件 | 作用 |
|---|---|
| `joint_flow.py` | 从 0.25 h 小 flow checkpoint 续训。`--mode frozen`：冻结 codec，只有 flow matching（对照臂）；`--mode joint`：codec 一起训练，加 $L_{AE}$、可选的 $L_{match}$ 和 KL。单卡。 |
| `frozen_flow_ddp.py` | 冻结 codec 的单机多卡 DDP 集中训练，带 EMA。第一轮的 `frozen_ddp_main` 用它。已知 bug：EMA 没有写进 checkpoint（修订 C）。保留它是为了复现第一轮，新实验请用 `stable_flow_ddp.py`。 |
| `stable_flow.py` | `StableObjectFlow`：在 `SmallObjectFlow` 上对 self 与 cross attention 做逐头 QK RMS 归一化；从旧权重热启动并在固定 batch 上校准新增增益。 |
| `stable_flow_ddp.py` | 稳定版集中训练：`StableObjectFlow`、warmup + cosine 学习率、完整 EMA、发散看门狗、定期 pin checkpoint。 |
| `object_views.py` | 只读诊断：把每个未来像素归给混色权重最大的 parent，与 MOVi 实例分割对比，以最近 anchor 划分为参照；输出指标、GIF 和轨迹图。 |
| `test_stable_flow.py` | `stable_flow.py` 的 CPU 测试：热启动、校准那一遍与旧模型一致、logit 有界。 |
| `memory_eval.py` | 只读：用训练器自己的 `evaluate()` 给同一 checkpoint 在训练集（`pilot_train1024` 前 32 段）、pilot32、confirmation64 上打分，区分记忆与 heldout 生成。 |
| `anneal_branch.py` | 从稳定版 run 的某个 checkpoint 分支出学习率退火 run，继承 flow、EMA、优化器和数据流。 |
| `long_rollout.py` | 只读：84 帧自回归续写。第 0 段用真实前缀，之后把生成的最后 4 帧经冻结 VAE 重编码为条件。只有第 0 段有真值。 |
| `cache_mirrored_features.py` | 把训练集水平镜像后，用同一个冻结 VAE 和编码函数生成特征缓存。 |
| `stable_flow_mirror_ddp.py` | 包装 `stable_flow_ddp.py`（不改动），训练集加上镜像副本；评测只用原始 validation。 |
| `fetch_movi_segmentations.py` | 按 byte range 读取官方 MOVi-E 记录，核对 SHA 与 RGB 逐像素相等，只保留实例分割与少量实例字段。 |
| `oracle.py` | oracle codec：25 个 GT 实例槽加 225 个背景 parent，按掩码取像素的编码器，TrajectoryDecoder，存在性 logit；`OracleObjectFlow` 加每槽前缀 token。 |
| `oracle_train.py` | oracle 的 DDP 训练，`--stage codec` 与 `--stage flow`，控制文件与预算规则同上。 |
| `oracle_eval.py` | oracle 评测：`cohort`、`views`（物体所有权与轨迹）、`rollout`（用渲染帧与渲染所有权续写）。 |
| `fetch_movi_extra_features.py` | 流式读取官方 MOVi-C/D/F 的 train shard，只保存冻结 VAE 特征，不落盘 RGB；不读 test 或 validation。 |
| `make_scratch_branch.py` | 写出任意宽度/深度的未训练 step-0 起点，让 `stable_flow_ddp.py` 从头训练新规模。 |
| `stable_flow_more_data_ddp.py` | 包装 `stable_flow_ddp.py`（不改动），训练集加上 MOVi-C/D/F 特征，可选再加镜像；评测只用 MOVi-E validation。 |

## 被导入的模块（src/code_base_v4/）

| 文件 | 作用 |
|---|---|
| `codec_research.py` | `Codec`（后验编码器 + 解码器 + 渲染）、`CacheBank`（读数据 manifest 与 VAE 特征缓存）、cohort 选择、query 采样、checkpoint 与哈希工具；也是阶段 1 codec 训练的入口。 |
| `codec_flow_research.py` | 冻结 codec 上的条件 flow：`ConditionalObjectFlow`、`load_codec`、flow matching 损失、前缀条件、评测；也是阶段 2 小 flow 训练的入口。 |
| `small_object_flow.py` | `SmallObjectFlow`：8 层、宽 384、6 头、250 个 slot embedding 的随机初始化 flow。 |
| `model.py` | 基础模块：`ObjectPosterior`、`FlowBlock`、`ObjectFlow`、早期的 `CompositeDecoder` 等。 |
| `trajectory_codec.py` | `TrajectoryDecoder`：持续存在的 parent、Bézier 轨迹与仿射、局部 Gaussian 子核、归一化密度混色。 |
| `trajectory_codec_3dtex.py` | 几何同上，纹理改为 $(u,v,t)$ 局部 RBF 的解码器变体。实际用哪个变体由 codec checkpoint 的 `variant` 配置决定。 |
| `eval_validation.py` | 评测指标、固定噪声（`clip_noise`）、按 clip 配对汇总与 bootstrap。 |
| `diagnose_persistence.py` | 持续性诊断与 GIF 输出（`time_metrics`、`save_gif`）。 |
| `wan_model.py` | 位置特征 `position_features`，以及 Wan 适配基线 `WanObjectFlow`。只有 Wan 路线会加载 diffusers 的 Wan 权重。 |
| `wan_token_adapter.py` | Wan 基线的对象 token 适配器。被 `codec_flow_research.py` 导入，小 flow 路线不使用。 |
| `test_small_object_flow.py` | 小 flow 契约的 CPU 测试，不衡量视频质量。 |

## 运行

CPU 测试：

```bash
cd src && python test_stable_flow.py
```

```bash
cd src/code_base_v4 && python -m unittest test_small_object_flow
```

训练和诊断脚本的必需入参（以 `--help` 为准）：

- `--data`：MOVi-E 数据快照目录，需含 `manifest.json`；
- `--cache`：由该快照生成的冻结 Wan2.2 VAE 特征缓存，需含 `protocol.json`，并与 manifest 哈希匹配；
- `--codec-checkpoint` 与 `--codec-sha`：C2 codec；
- `--cohorts`：showcase4 / pilot32 等固定划分；
- `--init-flow`（`joint_flow.py`）或 `--branch-from`（两个 DDP 脚本）及对应 SHA：起点 flow checkpoint；
- 预算：目标训练时长、占用上限和 `--wall-deadline-unix`，都是必填，没有默认的无限训练。

这些数据、缓存、checkpoint 和 cohort 文件都不在本仓库里；生成数据快照与 VAE 缓存的脚本也还没有放进来。

运行中的控制：在输出目录放 `REQUEST_PAUSE` 会保存后退出，放 `REQUEST_STOP` 会保存、跑最终 pilot32 评测后退出。

## 修改约定

- 训练脚本把自身源码的 SHA256 写进 checkpoint，resume 时会校验。改名或修改这些文件后，旧 checkpoint 不能直接 resume。
- `code_base_v4/` 是不可变快照的副本。要改其中的模块，请新建 `code_base_v5/` 之类的新版本，并记录与 v4 的差异。
