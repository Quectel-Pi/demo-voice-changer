# Shortest start-up and demo guide

> 中文版见 [DEMO_GUIDE_zh.md](DEMO_GUIDE_zh.md)。

This guide covers the source-tree demo in an environment that already has a working Qt, NumPy, sounddevice/PortAudio and Rubber Band 3.3 setup.
It is not a fresh-board installation acceptance or a third-party distribution licence conclusion; dependency boundaries are described in the [README](../README.md).
Local start-up and configuration regression has been verified; this build has also been deployed standalone on a Pi M2 and validated with a real window start and limited recovery (see the board start-up evidence).
On 2026-11-04 the desktop-icon extension (`ding@rastersoft.com`) was ACTIVE and a trusted entry was written, so desktop icons can be double-clicked;
this round did not repeat the manual double-click check, and screenshots/automated clicks are unavailable on this board. Fresh installation, the `.deb` packaging and formal performance acceptance are still outstanding.

## 1. Generate the desktop entry

Run this from the demo directory and select the verified Python explicitly; the output's parent directory must exist, and an existing file is not overwritten:

```bash
.venv/bin/python -B scripts/create_desktop_launcher.py \
  --python "$PWD/.venv/bin/python" \
  --output /absolute/path/to/your-desktop/voice-changer-demo.desktop
```

Replace the example output path with **your actual desktop directory**; do not assume it is called `Desktop`. The user applications menu directory also works.
The generator only writes the requested file: it does not install dependencies, query devices, start the program or modify the system.
Regenerate it after moving the source tree or the interpreter; an interpreter path containing `%` is rejected explicitly so GLib does not mis-parse it.
Spaces, quotes, `$` and `%` in the source path have been verified with real GLib argument probes.

With desktop icon support, double-click the generated entry (this board's GNOME 48 has no icon extension, so use the applications menu or `gio launch <file>`);
the desktop may ask you to allow launching or trust the file — confirm it yourself; the trust flag is never modified automatically. A real `gio launch` start, the 640×440 window and a normal exit were verified on the Pi M2.
The entry keeps a terminal open for error messages; the launcher script only waits for the native application, and audio still lives in that single application process.
Failures (missing Qt, a plugin exiting abnormally) show the exit status in the terminal and wait for Enter; this is not a background service.
No particular working directory or manual `PYTHONPATH` is required. Existing Rubber Band environment variables are passed through to the application.
No icon resources are generated; the system microphone icon is used, and a missing icon theme does not prevent start-up.

## 2. Running a demo

1. Wear wired headphones and check that the current system volume is safe; after start-up the voice is original, the gain is 0 dB, and all audio activity is stopped.
2. Click "Refresh devices" and confirm the input/output devices. Without saved history, enter 48000/512/1 and the peak limit you have confirmed yourself; the current demo path uses 0.05.
3. Click Listen and confirm the output, speak and switch through the nine presets; adjust pitch/formant/effect amount as needed. Manual effect changes become Custom; Reset keeps the gain.
4. Click Record, keep speaking and switch a few effects; you can stop monitoring and keep recording, or stop recording and keep monitoring.
5. After stopping, audition the original and the processed track separately. Playback releases capture first and does not re-apply effects or gain, and it does not restore monitoring when it ends.
6. Save to a new take directory, then exit normally; dual-track/parameter and alignment details are in the recording notes in the development workspace.

Recording alone does not need an output device; playback alone does not need an input device. Keep about one second of silence after speaking so the tail of the take is not cut.
On device/processing/disk errors stop first and read the error; do not raise the system volume and do not kill an existing RVC process.

## 3. Limited recovery

- Configuration lives in `$XDG_CONFIG_HOME/voice-changer/preferences.json`, or `~/.config` when unset.
  It records only the device names/backend names that were actually used successfully, the four audio values and the last successfully saved parent directory; device indices are not recorded.
- Valid history fields are pre-filled only and **never applied automatically**; without valid history they stay empty.
  Devices are matched only after the user refreshes; a missing, ambiguous or under-channeled device asks for a new choice and never falls back to a default device.
- Effects/gain/activity are not restored; nothing is probed, captured, played back or re-opened automatically; every output still needs confirmation.
- Reads, writes and directory checks happen on the control thread; during operation only memory is updated, and the configuration is written atomically after the device is released.
  Without history the file is only read and no configuration directory is created; a corrupt configuration or unusable directory is reported explicitly and the original file is never deleted automatically.
- If writing the configuration fails, the audio files and the pending snapshot are kept; fix the path and stop or exit to retry. If exit fails, a "retry exit" window is kept.

Tests and unverified boundaries are in the acceptance index and the board start-up records (the real device is `pipewire`/48000/512/mono/0.05).

## 4. Virtual microphone (v1)

Route the processed voice to other applications (meeting/recording/live tools):

1. Select `vcmic` as the output device (shown as `vcmic [#4, ALSA]`); keep the real microphone as input (for example `pipewire`).
2. Click "Listen"; the application validates the configuration and creates the PipeWire nodes without an extra safety-confirmation dialog:
   `voice-changer-sink` (input) and `voice-changer-mic` (for other applications to use).
3. In the other application, choose **Voice Changer Virtual Mic** as the microphone.
4. Stopping monitoring or exiting the application removes the nodes; do not leave a `pw-loopback` running when it is not needed.

Notes:

- A running PipeWire user session and `pw-loopback`/`pw-dump` are required; when missing, the UI states the reason and the virtual microphone entry does not appear.
- **It will not silently succeed**: with the target node absent, the ALSA compatibility layer "opens successfully" and discards every sample; this implementation validates the node with `pw-dump` and reports an error when it is not ready.
- The output has a single-consumer semantic: choosing `vcmic` sends the voice into the virtual microphone and no longer auto-plays it to speakers/headphones; to hear it yourself at the same time, use the other application's monitoring or switch back to a real output device.
- Bluetooth adds a lot of latency and is not recommended for the virtual microphone path.

## 5. One-step deployment (optional)

```bash
.venv/bin/python -B scripts/deploy_demo.py \
  --destination ~/voice-changer-$(date +%Y%m%d) \
  --python /usr/bin/python3 \
  --install-audio \
  --desktop-entry ~/.local/share/applications/voice-changer.desktop
```

The script does exactly four things: copy the source (excluding caches/recordings/virtual environments), create a `--system-site-packages` venv, install the audio dependencies by hash, and write the desktop entry. **It does not start audio, change PipeWire or the volume, touch system services or install system packages.**
`--dry-run` only prints the steps; `--no-venv` only copies the source; `--archive x.tar.gz` also saves a distribution archive.

It takes about 1–2 minutes (faster with cached dependencies). After deployment:

```bash
~/voice-changer-YYYYMMDD/venv/bin/python -B -m voice_changer --version
```

## 6. Known limitations (demo version v1)

- Latency is recorded but not gated: the scheduling buffer, actual DSP reserve and stream-reported values are in this round's measured report; the reserve depends on the configuration and cannot be reused from old runs;
  **end-to-end acoustic latency has not been measured** — do not claim specific latency numbers publicly.
- The A2 physical chain passed an hour of CPU/error checks; one-hour listening, multi-person subjective quality, audible parameter response and other sound cards/Bluetooth/USB remain untested.
- Screenshots and automated clicks are unavailable on this board; double-clicking the desktop icon still needs manual verification.
