# LLVC on RK3576 NPU

> 中文版见 [RKNN_LLVC_zh.md](RKNN_LLVC_zh.md)。

## Status and route

The desktop/board default remains the validated ONNX Runtime CPU path. RKNN is an explicit experimental backend. The board's Lite2/RKNPU stack passed a synthetic-input smoke test using the vendor's RK3576 YOLOv5 sample, but LLVC has not passed host-simulator accuracy validation or been run on the board.

The fixed LLVC graph builds with Toolkit2 2.5.0 and 2.3.2, but its host-simulator outputs do not satisfy the required `rtol=0.001`, `atol=0.0001` comparison. On zero audio/zero state, the first confirmed encoder mismatch is DCC0 `LayerNormalization` (converted to ExNorm; max absolute error `0.00567245`). Fixed-random and alternating-boundary inputs diverge earlier in the down-convolution state chain. Toolkit2 2.5.0 rejects `float_dtype="float32"` and `float_dtype="tfloat32"` for this target; `bfloat16` also fails the same DCC0 check (max absolute error `0.0511055`), and routing ExNorm to CPU did not change its result. The converter therefore checks the built graph in the host simulator before exporting and refuses to create an artifact when any output or carried state fails. The runtime also rejects manifests without this passing result. Do not use prior diagnostic/acceptance artifacts as deployment models.

RKNN is not an ONNX Runtime execution provider. The intended path is:

1. Convert only the fixed, SHA-256-pinned 104 ms LLVC graph on a supported host with `rknn-toolkit2`.
2. Load the resulting `.rknn` model on RK3576 through `rknn-toolkit-lite2` (or its RKNPU Runtime C API equivalent) and the board's RKNPU driver.
3. Keep the 48 kHz resampling and bounded worker pipeline unchanged. Each 16 kHz model invocation receives `audio`, `enc`, `dec`, `out`, and `pre`; `next_enc`, `next_dec`, `next_out`, and `next_pre` become the next invocation's states. A new converter starts with all-zero state.

The official [RKNN-Toolkit2 repository](https://github.com/airockchip/rknn-toolkit2) lists RK3576 as supported and describes host-side conversion followed by board-side RKNN Runtime/Lite inference. Its published README currently lists Python through 3.12 for Toolkit2, while this GUI must use Debian's Python 3.13. Therefore, do conversion in a separate supported host environment; verify the exact Lite2 package/ABI and driver compatibility on the board before installing anything into the GUI environment. A successful host conversion does not prove that Lite2 imports under Python 3.13.

## Operator and state compatibility

The accepted ONNX graph is fixed-shape, float32 and 104 ms. Its ordered input shapes are:

| Input | Shape |
|---|---|
| `audio` | `[1, 1, 1696]` |
| `enc` | `[1, 512, 510]` |
| `dec` | `[1, 2, 13, 256]` |
| `out` | `[1, 512, 4]` |
| `pre` | `[1, 1, 24]` |

Outputs are `converted`, `next_enc`, `next_dec`, `next_out`, `next_pre`; the four state outputs match their corresponding state input shapes. Previous ONNX validation established exact state continuity and repeatable zero-state reset. It did not test RKNN.

Operator compatibility must be established by the selected Toolkit2 version's actual `load_onnx`/`build` result and conversion report. The Toolkit2 v2.5.0 notes mention GRU support in some configurations, but do not establish support for this graph's exact GRU/layout/shape combination. Do not treat successful graph loading as proof of full NPU placement: inspect the report for unsupported operators or CPU fallback, then compare every audio/state output against ONNX. Quantization is disabled for the first conversion; any later quantized artifact is a separate model requiring its own accuracy and streaming validation.

## Conversion

Use a separate x86_64/Python 3.10 environment and the matching official `rknn-toolkit2` wheel from the upstream v2.5.0 release tree (`rknn-toolkit2/packages/x86_64`). Install the host-only graph preparation dependencies from `requirements-rknn-convert.txt` in that same isolated environment. Do not install conversion dependencies into a system, desktop application, or board GUI environment.

```bash
.venv-rknn/bin/python -m pip install -r requirements-rknn-convert.txt
.venv-rknn/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.3.1+cpu
.venv-rknn/bin/python -m pip install /path/to/official/rknn_toolkit2-2.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
.venv-rknn/bin/python scripts/convert_llvc_rknn.py \
  /path/to/llvc-stateful-8.onnx \
  /path/to/llvc-stateful-8.rknn
```

The conversion script refuses a different source graph, another target, quantization, and overwriting existing output. Before Toolkit2 sees the graph, it fixes the known input shapes, simplifies it with ONNX Simplifier, and replaces only constant-index `ScatterND` updates that form a complete contiguous region with equivalent `Slice`/`Concat` operations (or a full-update `Reshape`/`Identity`). It then requires ONNX Runtime equivalence over continuous zero, random, alternating-boundary, and positive/negative boundary inputs, carrying all four states separately through every block. The input/output tensor names, order, shapes, and float32 types are checked. Any non-canonical scatter pattern, simplifier failure, or numeric mismatch aborts conversion.

Before export, the converter initializes the Toolkit2 host simulator and checks all five outputs over deterministic silence, random, and boundary inputs, carrying all four states unchanged for three blocks. It applies the same fixed tolerance and aborts conversion on an invalid shape, dtype, non-finite value, simulator error, or numerical mismatch. The current Toolkit2 builds fail this gate, so no artifact is produced until the precision discrepancy is resolved.

The `.rknn.json` sidecar binds the original source hash and artifact hash and records the prepared graph hash, transformation/dependency versions, operator counts, ONNX Runtime equivalence results, and host-simulator equivalence results. Keep it with the model. Review all Toolkit2 warnings and placement details; conversion errors are fatal.

## In-memory board verification

The following probe imports neither the GUI nor `sounddevice`, generates deterministic float32 tensors in memory, never saves or plays generated audio, compares continuous output and all four carried states, replays a three-block sequence from zero state, and exercises 12 model blocks through the bounded worker while forcing its output buffer to silence:

```bash
.venv-board/bin/python -B scripts/bench_llvc_rknn.py \
  /path/to/llvc-stateful-8.onnx \
  /path/to/llvc-stateful-8.rknn \
  --warmup 4 --blocks 128 --rtol 0.001 --atol 0.0001
```

The initial numeric gate reuses the established ONNX-equivalence tolerance. A nonzero exit means at least one audio/state comparison failed; do not loosen it without reviewing the per-tensor maximum errors and the resulting voice-conversion quality. The JSON reports CPU and RKNN mean/p95/max inference time and the number of blocks above the 104 ms model budget. These are graph-only timings, not end-to-end audio latency or a sustained device test.

Before calling the RKNN route usable, require all of the following:

1. `rknn-toolkit2` builds the exact pinned ONNX graph for `rk3576` without unsupported compute operators or unintended CPU fallback.
2. The board's RKNPU runtime loads the artifact in the Python 3.13 environment and runs the script with the NPU enabled.
3. All five outputs are float32, finite, correctly shaped and ordered; continuous audio and every state pass the comparison gate. Repeating from zero states must reproduce the same sequence.
4. NPU inference p95 and max remain below 104 ms with thermal/frequency conditions recorded. The probe's worker smoke must process all 12 model blocks without queue errors while its output is forced to zero; then extend the duration and monitor thermal/clock conditions before any device test.

Only after these gates should a separate real microphone/speaker test be proposed. It requires the user's explicit consent every time. Never enable core dumps. If the GUI is later launched for a separately authorized check, set `XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.05I0Y3`; startup must remain idle until the user explicitly starts audio.

## 2026-11-11 board probe

After explicit user authorization, the official `rknn-toolkit-lite2` 2.5.0 aarch64 CPython 3.10 wheel was SHA-256 verified and installed in `~/voice-changer-rknn-lite2-acceptance-20261111-py310`, an isolated Python 3.10.22 environment. The application's Python 3.13 environment was not modified; the official Lite2 package directory currently publishes cp310/cp311/cp312 wheels, not cp313. The `rknn-toolkit2` 2.5.0 arm64 cp310 wheel was also checksum-verified and installed in that isolated environment with its conversion dependencies.

The Lite2 runtime loaded `/usr/share/model/yolov5/RK3576/yolov5s-640-640.rknn`, initialized `NPU_CORE_0` successfully and returned three output tensors for a synthetic zero image: `(1,255,80,80)`, `(1,255,40,40)`, `(1,255,20,20)`. Runtime reported librknnrt 2.3.2 and driver 0.9.8. This confirms the board NPU is available; it does not validate LLVC.

The pinned LLVC 104 ms ONNX graph was then passed to Toolkit2 for conversion. Build logs repeatedly warned that the `convert_gather_to_slice_reshape` rule on `pre_gather` may enter an infinite loop and warned that converting the `ScatterND` nodes expands them into 510-way Split/Concat graphs. After about seven minutes, available memory fell below the 500 MiB safety floor (about 418 MiB). The conversion was stopped, its temporary directory and three generated `check*.onnx` diagnostic graphs were removed, and no RKNN artifact was exported. The conversion script now runs Toolkit2 from its temporary directory so future diagnostic graphs are cleaned with temporary files. Do not retry the same conversion unchanged. The LLVC operators, NPU placement, state/audio equivalence and inference speed therefore remain unverified; investigate an exporter graph rewrite that removes or simplifies the problematic Gather/ScatterND patterns, then repeat conversion only with adequate memory headroom.

Core dumps remained disabled throughout. No GUI or real audio devices were started during the NPU probe.
