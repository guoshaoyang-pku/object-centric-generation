# src

两层代码：

- `src/*.py`：2026-10-27 第一轮扩大训练与稳定化的代码，从原项目的 `research/20261027_claude/` 原样复制。
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
