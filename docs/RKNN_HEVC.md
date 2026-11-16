# HEVC voice-changer models on the RKNN / NPU real-time path

> 中文版见 [RKNN_HEVC_zh.md](RKNN_HEVC_zh.md)。

This document records the process, gates and measured data for converting the three public HEVoicechanger female-voice ONNX models (UI names: Natural Female / Bright Female / Gentle Female) into RKNN packages that run on the local RK3576 NPU, and for integrating them into the **AI voice** real-time page.

Status (2026-11-14): **experimental path; start/stop runs on the board, numeric accuracy and latency have host and board data; audio quality and end-to-end acoustic latency still need the user's listening acceptance**. It is not an ONNX Runtime provider, not a CPU fallback, and has no original-voice passthrough.

## 1. Summary first

| Item | Result |
| --- | --- |
| Board NPU inference (11264-sample window) | Natural Female 113.7 ms / Bright Female 108.0 ms / Gentle Female 114.7 ms (p95 ≤ 115.2 ms) |
| Audio chunk | 3072 samples @ 16 kHz = 192 ms; 9216 frames @ 48 kHz on the device side |
| Real-time factor | 0.562–0.597 (about 40% headroom) |
| Converted graph vs original ONNX | max_abs ≤ 3.2e-7, SNR 108–120 dB (Silence/Random/Boundary/Speech) |
| RKNN simulator vs converted ONNX | speech SNR 63.2 / 48.5 / 58.1 dB; random ≥ 43.1 dB |
| Board device vs original ONNX (same 11264 window) | correlation 0.99998–1.0; max_abs ≤ 4.3e-3; all finite (2.5.0 build) |
| Quantization | no int8: measured speech SNR 5.5 dB, below the gate. The models are fp16 |
| Added latency | 2 conversion blocks of start-up reserve (384 ms; a single block underruns after about 0.4 s) plus device/resampling buffers, about 0.4–0.6 s; no acoustic measurement |
| Resources | each `.rknn` is about 44 MB; reserve space in `~/` on the board; the NPU uses CORE_0 only |

## 2. Runtime constraints

- Board NPU stack: `librknnrt 2.3.2`, driver 0.9.8. Loading a model built with 2.5.0 prints
  `RKNN Model version: 2.5.0 not match with rknn runtime version: 2.3.2`, but inference is correct in practice.
- The desktop GUI uses Python 3.13; the official `rknn-toolkit-lite2` has no 3.13 wheel, so the NPU session runs in a
  Python 3.10 helper process and GUI/helper communicate over a binary pipe. The helper interpreter is set by
  `VOICE_CHANGER_RKNN_PYTHON` (on the board: `~/voice-changer-rknn-lite2-acceptance-20261111-py310/bin/python`).
- If the variable is unset, the interpreter is missing, the helper script is missing or the model manifest fails validation, **the start button stays disabled and the UI shows the specific reason**.
- If the helper crashes or returns non-finite samples, the real-time engine reports a fault and stops; it never falls back to CPU, ONNX, the original voice or DSP.

## 2.1 Why the models must be compiled with rknn-toolkit2 2.5.0 (a 2.3.2 compiler defect)

After the first deployment on 2026-11-14, listening on the board showed obvious mechanical artifacts, noise and dropouts. Investigation conclusion: **the 2.3.2 release of `rknn-toolkit2` generates wrong NPU code for this network on RK3576**, while the simulator (CPU fp32) and ONNX Runtime are both completely correct.

Evidence summary:

- On the same 11264-sample window, the correlation between the device output and ONNX was only 0.012–0.016; the output was nearly constant (the first and last samples were always `3.05e-5`), failing every accuracy gate.
- Layer-by-layer localisation put the first error at the first channel-mixing convolution after the encoder's `dcc_0` LayerNorm/Relu; that convolution alone builds and runs correctly on the device; 1×1 convolution, Transpose, MatMul, Slice/Concat, LayerNorm chains and other single operators are all correct, and the simulator was correct throughout. In the same compiled artifact the convolution's input tensor reads out correctly as a model output while the convolution result is wrong, which points at NPU compilation/execution rather than numeric range.
- Rewriting the operator form (MatMul, 2D reshape, inserted Reshape/identity operators), changing `optimization_level` (0/1/2) and switching `rknn-toolkit-lite2` versions (2.3.2/2.5.0) all produced byte-identical wrong output, ruling out operator form, fusion, optimisation level and lite version.
- Recompiling the same subgraph and the full model with `rknn-toolkit2==2.5.0` made the board output correct immediately (corr 1.0); 2.5.0-built models run normally on the board's 2.3.2 runtime (with only the version-mismatch warning).
- The conversion script therefore requires `rknn-toolkit2>=2.5.0`, and the application-side manifest check also requires `toolkit_version>=2.5.0`, so a wrong package produced by 2.3.2 cannot be mistaken for a usable model.

This is a toolchain defect, not something this repository can fix; it is recorded here so the trap is not hit again.

## 3. From ONNX to a usable RKNN package (host side)

Script: `scripts/convert_hevc_rknn.py` (runs on the host; conversion does not happen on the board). Dependencies:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.0+cpu
python -m pip install rknn-toolkit2>=2.5.0 "setuptools<80" onnx==1.16.1 onnxsim onnxruntime
```

`onnx==1.16.1` is a hard requirement (2.x removed `onnx.mapping`); `setuptools<80` provides the `pkg_resources` the toolkit needs; `torch` is only a toolkit dependency and is not used at inference.

Conversion steps:

1. Fix `audio_in` to `[1,1,11264]` (8192 history + 3072 chunk), run `onnx.shape_inference` then
   `onnxsim.simplify`; this reduces 3523 nodes to 516.
2. Rewrite the 20 constant-index `ScatterND` nodes as Slice/Concat/Reshape (the RKNN compiler rejects the original pattern). After the rewrite, compare with ONNX Runtime sample by sample; the four cases show max_abs ≤ 3.2e-7.
3. `rknn.config(target_platform="rk3576", optimization_level=3,
   disable_rules=["convert_rs_add_rs_to_rs_gather_elements"])`, `build(do_quantization=False)` to obtain an fp16 model (the GatherElements rule must be disabled explicitly, otherwise the build fails).
4. On the host simulator, compare the RKNN output with the rewritten ONNX: speech/noise SNR must be ≥ 40 dB and max_abs ≤ 1e-2 for every case; a failing case deletes the artifact and exits with an error.
5. Generate `<model>_rknn.rknn` and the sidecar `<model>_rknn.rknn.json` with the source ONNX SHA-256, build versions, window layout, simulator accuracy and per-case details.

Example:

```bash
cd demos/41-voice-changer
.venv/rknn232/bin/python scripts/convert_hevc_rknn.py --list   # list the three known source models
.venv/rknn232/bin/python scripts/convert_hevc_rknn.py --output-dir .venv/rknn-packages \
  --reference-audio /path/to/speech.wav \
  .venv/models-hevc/female_voice_hfg_new.onnx \
  .venv/models-hevc/female_voice_old_hq.onnx \
  .venv/models-hevc/female_voice_old.onnx
```

The script only accepts source SHA-256 values that belong to the three known public HEVoicechanger graphs; unknown weights are rejected so that an arbitrary `.rknn` cannot masquerade as a real-time model. The repository does not distribute these 41 MB ONNX or 132 MB RKNN weights (GPL-3.0); local artifacts live under the gitignored `.venv/`.

## 4. Model package and runtime validation (application side)

`src/voice_changer/ai/hevc_rknn.py` is the application-side backend; `scripts/hevc_rknn_worker.py` is the board-side helper process.

The manifest must satisfy: format `voice-changer-hevc-rknn-v1`, `target_platform=rk3576`,
`toolkit_version≥2.5.0`, `frames=11264`, `history=8192`, `chunk=3072`,
`model_sha256` matching the file, `source_onnx_sha256` belonging to a known HEVC graph, and `simulator_snr_db ≥ 40`.
If any item fails, the model only shows an "unavailable" reason and cannot enter the real-time chain.

Helper protocol (binary stdin/stdout, `struct "<4sI"`):

```
request: b"HEV1" | u32 sample_count | sample_count * float32 (full 11264-sample window)
ready:   b"HVRD" | u32 0
reply:   b"HEV1" | u32 3072 | 3072 * float32 (model output for the newest chunk)
error:   b"HVER" | u32 byte_length | utf-8 text (then exit)
```

The count in the request header is a **sample count**, the length in the error header is a **byte count**. The helper is stateless: every received full 11264-sample window is inferred once and the last 3072 samples are returned. The GUI-side converter keeps 8192 samples of history, the stateful 48 kHz↔16 kHz resampling, the 192 ms chunking, the start-up reserve and the final limiter; any resampling underrun raises an error.

## 5. Board measurements and acceptance

After building on the host, copy the three `.rknn` files and their manifests to a board directory (the deployment directory is `models/`), set
`VOICE_CHANGER_RKNN_PYTHON`, select a model on the AI voice page (the buttons show the corresponding Chinese names) and start after choosing the devices.

Verified (2026-11-14, real window):

- All three models are recognised by the application as `hevc_rknn`, the manifest check passes and the start button is enabled.
- After the compiler fix, the outputs of the three 2.5.0 packages on the board correlate 0.99998–1.0 with the original ONNX, max_abs ≤ 4.3e-3.
- The board NPU benchmark over 20 inferences averages 108.0–114.7 ms (11264 window, CORE_0), all outputs finite.
- In the real window, clicking Yes enters "NPU live conversion running", the helper stays alive with no underruns for 55 s; Stop exits the helper and returns to "not started (NPU experiment)".
- The start-up reserve must be 2 conversion blocks (384 ms): with only 1 block (192 ms) the real window reports `RVC output queue underrun` after about 0.4 s, because in-app conversion including resampling takes about 137–155 ms per block.
- The status line is shown in one language matching the UI language (Chinese "NPU 实验…" / English "NPU live conversion running") to avoid truncation.
- Unit/integration tests cover the manifest gates, the helper protocol, converter chunking, factory configuration, the start-up reserve and the UI start path.

Still to accept:

- Audio quality (different quality versions of the same public female voice can only be judged by listening; interface and accuracy checks are not quality).
- End-to-end acoustic latency and long-run jitter (192 ms chunks plus start-up reserve; this path is not low-latency).
- Multi-model concurrency and stability under board memory/load.
- The boundary between GPL-3.0 weights and repository release artifacts (do not bundle the weights into the repository or a release package).

## 6. Difference from the LLVC NPU attempt

The LLVC conversion recorded in `docs/RKNN_LLVC.md` is blocked at the compiler/host-simulator stage; the HEVC graphs build after the `ScatterND` rewrite and disabling the GatherElements rule, and pass on both the fp16 simulator and the board NPU. Neither is an ONNX Runtime NPU provider; both models must be converted in advance.
