# HEVoicechanger 模型的 RKNN / NPU 实时链路

> 英文版见 [RKNN_HEVC.md](RKNN_HEVC.md)。

本文记录把三个公开 HEEVoicechanger 女声 ONNX（界面名：自然女生 / 清亮女声 / 柔和女声）转换成本机
RK3576 NPU 可运行的 RKNN 包，并接入“AI 变声”实时页的过程、门限与实测数据。

状态（2026-11-14）：**实验链路，已在板端跑通启动/停止，数值精度和延迟有本机与板端数据；音质与
端到端声学延迟仍需用户试听验收**。不是 ONNX Runtime provider，不是 CPU 回退，也没有原声直通。

## 1. 结论先读

| 项目 | 结果 |
| --- | --- |
| 板端 NPU 推理（11264 样本窗口） | 自然女生 113.7 ms / 清亮女声 108.0 ms / 柔和女声 114.7 ms（p95 ≤ 115.2 ms） |
| 分块音频 | 3072 样本 @ 16 kHz = 192 ms；设备侧 9216 帧 @ 48 kHz |
| 实时系数 RTF | 0.562–0.597（约 40% 余量） |
| 转换后图形与原始 ONNX 的差异 | max_abs ≤ 3.2e-7，SNR 108–120 dB（Silence/Random/Boundary/Speech） |
| RKNN 模拟器 vs 转换后 ONNX | 语音 SNR 63.2 / 48.5 / 58.1 dB；random ≥ 43.1 dB |
| 板端设备 vs 原始 ONNX（同一 11264 窗口） | 相关系数 0.99998–1.0；max_abs ≤ 4.3e-3；全部有限（2.5.0 编译包） |
| 量化 | 未做 int8：实测语音 SNR 5.5 dB，不达门限。模型为 fp16 |
| 附加延迟 | 启动储备 2 个转换块（384 ms，单块储备实测会在约 0.4 s 后欠载）＋设备/重采样缓冲，约 0.4–0.6 s；未做声学测量 |
| 资源 | 每个 `.rknn` 约 44 MB；板端 `~/` 需预留空间；NPU 仅用 CORE_0 |

## 2. 运行时约束

- 板端 NPU 栈：`librknnrt 2.3.2`、驱动 0.9.8。加载 2.5.0 编译的模型时会打印
  `RKNN Model version: 2.5.0 not match with rknn runtime version: 2.3.2` 警告，但实测推理正确。
- 桌面 GUI 用 Python 3.13；官方 `rknn-toolkit-lite2` 没有 3.13 wheel，因此 NPU 会话放在
  Python 3.10 辅助进程里，GUI 与辅助进程用二进制管道通信。辅助进程解释器由
  `VOICE_CHANGER_RKNN_PYTHON` 指定（板端为 `~/voice-changer-rknn-lite2-acceptance-20261111-py310/bin/python`）。
- 未设置该环境变量、解释器不存在、辅助脚本缺失或模型包 manifest 校验失败时，**开始按钮保持禁用并显示具体原因**。
- 辅助进程崩溃或返回非有限样本时，实时引擎报故障并停止；不会回退到 CPU、ONNX、原声或 DSP。

## 2.1 为什么必须用 rknn-toolkit2 2.5.0 编译（2.3.2 编译器缺陷）

2026-11-14 首次部署后在板端试听出现明显机械感、杂音和断续。排查结论：**2.3.2 版
`rknn-toolkit2` 对这个网络在 RK3576 上生成了错误的 NPU 代码**，而模拟器（CPU fp32）与
ONNX Runtime 都完全正确。

证据概要：

- 设备输出与 ONNX 在同一 11264 窗口的相关系数只有 0.012–0.016，输出近似恒定（首尾样本恒为
  `3.05e-5`），无法通过任何精度门限。
- 逐层定位到编码器 `dcc_0` 的 LayerNorm/Relu 之后、第一个通道混合卷积处开始出错；该卷积单独
  建图在设备上正确，1x1 卷积、Transpose、MatMul、Slice/Concat、LayerNorm 链等单算子全部正确，
  模拟器全程正确。同一编译产物里，卷积的输入张量作为模型输出读出时正确，卷积结果却错误，
  说明问题在 NPU 编译/执行而非数值范围。
- 改写算子形态（MatMul、2D reshape、插 Reshape/恒等算子）、调整 `optimization_level`
  (0/1/2)、更换 `rknn-toolkit-lite2` 版本（2.3.2/2.5.0）都得到逐字节相同的错误输出；
  排除算子形态、融合、优化级别与 lite 版本因素。
- 用 `rknn-toolkit2==2.5.0` 重新编译同一个子图与完整模型后，板端设备输出立即正确
  （corr 1.0）；2.5.0 编译的模型可以在板端 2.3.2 runtime 上正常运行（只有版本不匹配警告）。
- 因此转换脚本强制要求 `rknn-toolkit2>=2.5.0`，应用侧 manifest 校验也要求
  `toolkit_version>=2.5.0`，防止把 2.3.2 生成的错误包当成可用模型。

此为工具链缺陷，不是本仓库可修复的问题；记录在此以免再次踩坑。

## 3. 从 ONNX 到可用的 RKNN 包（主机侧）

脚本：`scripts/convert_hevc_rknn.py`（主机侧运行，板端不转换）。依赖环境：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.0+cpu
python -m pip install rknn-toolkit2>=2.5.0 "setuptools<80" onnx==1.16.1 onnxsim onnxruntime
```

`onnx==1.16.1` 是硬要求（2.x 移除了 `onnx.mapping`）；`setuptools<80` 提供 toolkit 需要的
`pkg_resources`；`torch` 只是 toolkit 依赖，不会在推理中使用。

转换步骤：

1. 把 `audio_in` 固定为 `[1,1,11264]`（历史 8192 + 分块 3072），`onnx.shape_inference` 后
   `onnxsim.simplify`，3523 个节点降到 516 个。
2. 把 20 个常量索引的 `ScatterND` 改写成 Slice/Concat/Reshape（RKNN 编译器不接受原模式）。
   改写后用 ONNX Runtime 与原始图逐样本对比，四项用例 max_abs ≤ 3.2e-7。
3. `rknn.config(target_platform="rk3576", optimization_level=3,
   disable_rules=["convert_rs_add_rs_to_rs_gather_elements"])`，`build(do_quantization=False)`，
   得到 fp16 模型（GatherElements 规则需要显式关闭，否则构建报错）。
4. 在主机模拟器上把 RKNN 输出与改写后的 ONNX 对比：语音/噪声 SNR 必须 ≥ 40 dB，
   全部用例 max_abs ≤ 1e-2；不过门限会删除产物并失败退出。
5. 生成 `<模型>_rknn.rknn` 与同名 sidecar `<模型>_rknn.rknn.json`，包含来源 ONNX SHA-256、
   构建版本、窗口划分、模拟器精度与用例明细。

示例：

```bash
cd demos/41-voice-changer
.venv/rknn232/bin/python scripts/convert_hevc_rknn.py --list   # 列出三个已知来源模型
.venv/rknn232/bin/python scripts/convert_hevc_rknn.py --output-dir .venv/rknn-packages \
  --reference-audio /path/to/speech.wav \
  .venv/models-hevc/female_voice_hfg_new.onnx \
  .venv/models-hevc/female_voice_old_hq.onnx \
  .venv/models-hevc/female_voice_old.onnx
```

脚本只接受来源 SHA-256 属于已知三个公开 HEEVoicechanger 图；未知权重会被拒绝，避免把任意
`.rknn` 当成可实时模型。仓库不分发这 41 MB ONNX 或 132 MB RKNN 权重（GPL-3.0），本机产物放在
gitignored 的 `.venv/` 下。

## 4. 模型包与运行时校验（应用侧）

`src/voice_changer/ai/hevc_rknn.py` 是应用侧后端；`scripts/hevc_rknn_worker.py` 是板端辅助进程。

manifest 必须满足：格式 `voice-changer-hevc-rknn-v1`、`target_platform=rk3576`、
`toolkit_version≥2.5.0`、`frames=11264`、`history=8192`、`chunk=3072`、
`model_sha256` 与文件一致、`source_onnx_sha256` 属于已知 HEVC 图、`simulator_snr_db ≥ 40`。
任何一项不满足，模型只显示“不可用”原因，不允许进入实时链。

辅助进程协议（stdin/stdout 二进制，`struct "<4sI"`）：

```
请求: b"HEV1" | u32 sample_count | sample_count * float32（完整窗口 11264）
就绪: b"HVRD" | u32 0
回复: b"HEV1" | u32 3072 | 3072 * float32（最新分块的模型输出）
错误: b"HVER" | u32 byte_length | utf-8 文本（随后退出）
```

请求头里的计数是**样本数**，错误帧头里的长度是**字节数**。辅助进程无状态：每次收到完整
11264 样本窗口就推理一次，返回最后 3072 样本。GUI 侧的转换器负责维护 8192 样本历史、
48 kHz↔16 kHz 有状态重采样、192 ms 分块、启动储备与最终限幅，任何重采样欠载都会抛错。

## 5. 板端实测与验收

主机侧构建后，把三个 `.rknn` 与 manifest 复制到板端目录（部署目录 `models/`），设置
`VOICE_CHANGER_RKNN_PYTHON`，在 AI 变声页选择模型（按钮显示对应中文名），选好设备后开始。

已验证（2026-11-14，真实窗口）：

- 三个模型均能被应用识别为 `hevc_rknn`，manifest 校验通过，开始按钮可用。
- 修复编译器问题后，三个 2.5.0 包在板端设备上的输出与原始 ONNX 相关系数 0.99998–1.0，max_abs ≤ 4.3e-3。
- 板端 NPU 基准 20 次推理均值 108.0–114.7 ms（11264 窗口，CORE_0），全部输出有限。
- 真实窗口点击 Yes 后进入「NPU 实验实时变声运行中」，辅助进程存活，连续 55 s 无欠载；Stop 后辅助进程退出并回到「未启动（NPU 实验）」。
- 启动储备必须是 2 个转换块（384 ms）：只留 1 块（192 ms）时真实窗口约 0.4 s 即 `RVC output queue underrun`，因为应用内转换含重采样约 137–155 ms/块。
- 状态行按界面语言单语显示（中文「NPU 实验…」/英文「NPU live conversion running」），避免过长被截断。
- 单元/集成测试覆盖 manifest 门限、辅助进程协议、转换器分块、工厂配置、启动储备与 UI 启动路径。

仍需验收：

- 音质（同一公开女声的不同质量版本只能靠听；接口与精度检查不等于音质）。
- 端到端声学延迟与长时间抖动（192 ms 分块 + 启动储备，本链路不属于低延迟）。
- 多模型并发、板端内存与负载下的稳定性。
- GPL-3.0 权重与仓库发布物的边界（不要把权重并入仓库或发行包）。

## 6. 与 LLVC NPU 尝试的区别

`docs/RKNN_LLVC.md` 记录的 LLVC 转换已在编译器/host simulator 阶段受阻；HEVC 图在完成
`ScatterND` 改写、关闭 GatherElements 规则后可以构建，并在 fp16 模拟器和板端 NPU 上都通过。
两者都不是 ONNX Runtime 的 NPU provider，模型都必须预先转换。
