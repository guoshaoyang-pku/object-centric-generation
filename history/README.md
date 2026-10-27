# 2026-10-27 源码快照（历史）

这里是 2026-10-27 夜间实验实际使用的六版源码，按字节复制，按时间顺序各成一个提交。项目在这之前没有 git 历史，这几个提交就是代码演化的记录。这里只有源码，没有权重、数据、缓存、日志或图像。

| 目录 | 当夜职责 | 主要入口 |
|---|---|---|
| [code](code/) | v1：A/B 时空混合核与 C2 共享轨迹 codec；VAE 特征缓存、重建诊断；原 Wan 对象 flow 适配 | `codec_research.py`、`codec_flow_research.py` |
| [code_v2](code_v2/) | 局部三维纹理 C3，子核与 parent 数的容量对照 | `codec_research.py`、`trajectory_codec_3dtex.py` |
| [code_v3](code_v3/) | 冻结解码器，比较编码器继续训练与两层 mixer | `codec_research.py` |
| [code_v4](code_v4/) | 同一冻结 codec 上，Wan + LoRA 对比随机初始化的小型全局对象 flow。当前 `src/code_base_v4/` 就是这一版 | `codec_flow_research.py`、`small_object_flow.py` |
| [code_v5](code_v5/) | 规则时空 VAE patch 基线（96×192 打包，经冻结 VAE 解码），不是原始像素扩散 | `patch_flow_research.py`、`native_patch_codec.py` |
| [code_v6_pixel](code_v6_pixel/) | 原始 RGB64 像素基线（20 帧打包为 1280×192） | `raw_pixel_flow_research.py`、`raw_pixel_codec.py` |

## 校验

[archive_manifest.json](archive_manifest.json) 列出每个文件的 SHA256 和字节数、每版源码树的 SHA，以及对应运行的 protocol 绑定。公开前删掉了主机名和绝对路径，哈希值没有改动。

各子目录里的 `checksums.json` 和 `source_copy_manifest.json` 也是按字节保留的原件：前者可能仍列出未归档的 `changes.patch`，后者记录的是复制前的上游版本，所以不要拿它们当本版文件的当前校验表。各子目录 `README.md` 里的“等待 gate”“尚未启动”之类的字样是当时的准备状态，不是现在的实验状态。

## 复用

这些目录是源码归档，不是可以直接运行的安装包。要重跑，还需要原数据、feature cache、cohorts、冻结 codec checkpoint、必要的 Wan 权重或 raw-pixel 统计文件，并通过 runner 的 SHA/protocol 检查。修改源码后，请用新版本号和新输出目录；训练输出不要写进源码树。
