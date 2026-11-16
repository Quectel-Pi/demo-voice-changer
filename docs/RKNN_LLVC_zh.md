# LLVC 在 RK3576 NPU 上的运行尝试

> English: [RKNN_LLVC.md](RKNN_LLVC.md)。

## 状态与路线

桌面/板端的默认路径仍是已验证的 ONNX Runtime CPU 链路。RKNN 是显式的实验后端。板端的 Lite2/RKNPU 栈使用厂商的 RK3576 YOLOv5 样例通过了合成输入冒烟测试，但 LLVC 尚未通过主机侧模拟器的精度验证，也没有在板端运行过。

固定形状的 LLVC 图在 Toolkit2 2.5.0 与 2.3.2 上都能构建，但主机侧模拟器输出不满足要求的 `rtol=0.001`、`atol=0.0001` 对比。在零音频/零状态下，首个确认的编码器不一致出现在 DCC0 的 `LayerNormalization`（被转换为 ExNorm；最大绝对误差 `0.00567245`）。固定随机输入与交替边界输入在更早的下采样卷积状态链上就出现偏离。Toolkit2 2.5.0 在该目标上拒绝 `float_dtype="float32"` 与 `float_dtype="tfloat32"`；`bfloat16` 也无法通过同一个 DCC0 检查（最大绝对误差 `0.0511055`），把 ExNorm 放到 CPU 上执行也没有改变结果。因此转换脚本在导出前会在主机模拟器中检查构建出的图，任何输出或传递状态不通过就拒绝生成产物；运行时也会拒绝没有该通过结果的 manifest。不要用以往的诊断/验收产物当作部署模型。

RKNN 不是 ONNX Runtime 的执行提供者。预期路径是：

1. 只在受支持的主机上用 `rknn-toolkit2` 转换固定、SHA-256 锁定的 104 ms LLVC 图。
2. 在 RK3576 上通过 `rknn-toolkit-lite2`（或等价的 RKNPU Runtime C API）和板端 RKNPU 驱动加载生成的 `.rknn` 模型。
3. 48 kHz 重采样与有界 worker 流水线保持不变。每次 16 kHz 模型调用接收 `audio`、`enc`、`dec`、`out`、`pre`；`next_enc`、`next_dec`、`next_out`、`next_pre` 成为下一次调用的状态。新的转换器从全零状态开始。

官方 [RKNN-Toolkit2 仓库](https://github.com/airockchip/rknn-toolkit2) 将 RK3576 列为支持平台，并描述了主机侧转换后由板端 RKNN Runtime/Lite 推理的流程。其公开 README 当前列出的 Toolkit2 Python 支持上限为 3.12，而本 GUI 必须使用 Debian 的 Python 3.13。因此请在单独受支持的主机环境中转换；在把任何东西装进 GUI 环境之前，先验证确切的 Lite2 包/ABI 与板端驱动兼容性。主机转换成功并不证明 Lite2 能在 Python 3.13 下导入。

## 算子与状态兼容性

已接受的 ONNX 图为固定形状、float32、104 ms。其输入顺序与形状为：

| 输入 | 形状 |
|---|---|
| `audio` | `[1, 1, 1696]` |
| `enc` | `[1, 512, 510]` |
| `dec` | `[1, 2, 13, 256]` |
| `out` | `[1, 512, 4]` |
| `pre` | `[1, 1, 24]` |

输出为 `converted`、`next_enc`、`next_dec`、`next_out`、`next_pre`；四个状态输出与其对应状态输入的形状一致。此前的 ONNX 验证已确认状态连续性与可重复的零状态复位，但没有测试 RKNN。

算子兼容性必须以所选 Toolkit2 版本实际的 `load_onnx`/`build` 结果与转换报告为准。Toolkit2 v2.5.0 的说明提到某些配置下支持 GRU，但并未证明支持本图确切的 GRU/布局/形状组合。不要把图加载成功当作算子全部放到 NPU 的依据：检查报告中不受支持的算子或 CPU 回退，然后把每个音频/状态输出与 ONNX 逐一对比。首次转换关闭量化；之后任何量化产物都是独立模型，需要单独做精度与流式验证。

## 转换

使用独立的 x86_64/Python 3.10 环境，并从上游 v2.5.0 发布树（`rknn-toolkit2/packages/x86_64`）安装匹配的官方 `rknn-toolkit2` wheel。在同一隔离环境中安装 `requirements-rknn-convert.txt` 中的主机侧建图依赖。不要把转换依赖装进系统环境、桌面应用或板端 GUI 环境。

```bash
.venv-rknn/bin/python -m pip install -r requirements-rknn-convert.txt
.venv-rknn/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.3.1+cpu
.venv-rknn/bin/python -m pip install /path/to/official/rknn_toolkit2-2.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
.venv-rknn/bin/python scripts/convert_llvc_rknn.py \
  /path/to/llvc-stateful-8.onnx \
  /path/to/llvc-stateful-8.rknn
```

转换脚本会拒绝不同的来源图、其他目标平台、量化以及覆盖已有输出。在 Toolkit2 看到图之前，它先固定已知输入形状，用 ONNX Simplifier 化简，并只把构成完整连续区域的常量索引 `ScatterND` 更新替换为等价的 `Slice`/`Concat` 操作（或整体更新的 `Reshape`/`Identity`）。随后要求在连续零值、随机、交替边界、正/负边界输入上通过 ONNX Runtime 等价性检查，并在每个块中分别携带全部四个状态。输入/输出张量名、顺序、形状与 float32 类型都会被检查。任何非规范的 scatter 模式、化简失败或数值不一致都会中止转换。

导出前，转换脚本初始化 Toolkit2 主机模拟器，在确定性静音、随机与边界输入上检查全部五个输出，并在三个块中保持四个状态不变。它使用同一固定容差，遇到无效形状、dtype、非有限值、模拟器错误或数值不一致就中止转换。当前 Toolkit2 构建无法通过该门限，因此在精度差异解决前不会产出任何制品。

`.rknn.json` sidecar 绑定原始来源哈希与制品哈希，并记录预处理图的哈希、变换/依赖版本、算子数量、ONNX Runtime 等价性结果与主机模拟器等价性结果。请把它和模型放在一起。仔细检查 Toolkit2 的全部警告与算子放置细节；转换错误是致命的。

## 板端内存内验证

以下探测脚本既不导入 GUI 也不导入 `sounddevice`，只在内存中生成确定性 float32 张量，从不保存或播放生成的音频，比较连续输出与全部四个传递状态，从零状态重放三个块的序列，并通过有界 worker 跑 12 个模型块（同时把其输出缓冲强制置为静音）：

```bash
.venv-board/bin/python -B scripts/bench_llvc_rknn.py \
  /path/to/llvc-stateful-8.onnx \
  /path/to/llvc-stateful-8.rknn \
  --warmup 4 --blocks 128 --rtol 0.001 --atol 0.0001
```

初始数值门限沿用既有的 ONNX 等价容差。非零退出码表示至少一个音频/状态对比失败；在审查每张张量的最大误差与最终变声音质之前，不要放宽该门限。JSON 报告 CPU 与 RKNN 推理的均值/p95/最大值，以及超出 104 ms 模型预算的块数。这些只是图级计时，不是端到端音频延迟，也不是持续设备测试。

在宣称 RKNN 路线可用之前，必须全部满足：

1. `rknn-toolkit2` 能在没有不受支持的计算算子、也没有意外 CPU 回退的情况下为 `rk3576` 构建确切的锁定 ONNX 图。
2. 板端 RKNPU 运行时能在 Python 3.13 环境中加载该制品，并在启用 NPU 的情况下运行脚本。
3. 全部五个输出为 float32、有限、形状与顺序正确；连续音频与每个状态都通过对比门限。从零状态重复必须复现同一序列。
4. NPU 推理的 p95 与最大值保持在 104 ms 以下，并记录热/频率条件。探测脚本的 worker 冒烟必须在输出被强制置零的情况下处理全部 12 个模型块且无队列错误；随后在设备测试前延长持续时间并监控热/时钟条件。

只有通过这些门限后，才应考虑单独的真实麦克风/扬声器测试；每次都需要用户明确同意。永远不要开启 core dump。如果之后为单独授权的检查启动 GUI，请设置 `XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.05I0Y3`；启动后必须保持空闲，直到用户明确开始音频。

## 2026-11-11 板端探测

在用户明确授权后，官方 `rknn-toolkit-lite2` 2.5.0 aarch64 CPython 3.10 wheel 经 SHA-256 校验后安装到 `~/voice-changer-rknn-lite2-acceptance-20261111-py310`，这是一个隔离的 Python 3.10.22 环境。应用的 Python 3.13 环境未被修改；官方 Lite2 包目录当前发布 cp310/cp311/cp312 wheel，没有 cp313。`rknn-toolkit2` 2.5.0 arm64 cp310 wheel 也经校验后连同转换依赖安装到该隔离环境。

Lite2 运行时加载了 `/usr/share/model/yolov5/RK3576/yolov5s-640-640.rknn`，成功初始化 `NPU_CORE_0`，并对合成零值图像返回三个输出张量：`(1,255,80,80)`、`(1,255,40,40)`、`(1,255,20,20)`。运行时报告 librknnrt 2.3.2、驱动 0.9.8。这确认板端 NPU 可用，但不验证 LLVC。

随后把锁定的 LLVC 104 ms ONNX 图交给 Toolkit2 转换。构建日志反复警告 `pre_gather` 上的 `convert_gather_to_slice_reshape` 规则可能进入死循环，并警告转换 `ScatterND` 节点会展开成 510 路 Split/Concat 图。约七分钟后，可用内存降到 500 MiB 安全下限以下（约 418 MiB）。转换被停止，其临时目录与三个生成的 `check*.onnx` 诊断图被删除，未导出任何 RKNN 制品。转换脚本现在从临时目录运行 Toolkit2，以便将来的诊断图随临时文件一起清理。不要原样重试同一次转换。LLVC 的算子、NPU 放置、状态/音频等价性与推理速度因此仍未经验证；请先研究能移除或简化问题 Gather/ScatterND 模式的导出图改写，然后在内存余量充足时再重复转换。

整个过程 core dump 保持关闭；NPU 探测期间没有启动 GUI 或真实音频设备。
