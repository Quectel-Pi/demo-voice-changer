"""Opt-in, finite M0-A probe; not the production audio engine."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


SOURCE = "alsa_input.platform-es8389-sound.analog-stereo"
SINK = "alsa_output.platform-es8389-sound.analog-stereo"
SOURCE_CHANNELS = ("front-left", "front-right")

# Acoustic on/off markers for through mode: a downward sweep is nearly orthogonal
# to the upward 700-4000Hz test chirps under matched filtering.
MARKER_AMPLITUDE = 0.01
MARKER_SECONDS = 0.06
MARKER_SWEEP_HZ = (4000, 700)
MARKER_START_SECONDS = (0.25,)
MARKER_END_SECONDS_BEFORE_STOP = (0.5, 0.25)


def marker_waveform(rate):
    count = round(MARKER_SECONDS * rate)
    t = np.arange(count) / rate
    high, low = MARKER_SWEEP_HZ
    envelope = np.sin(np.pi * np.arange(count) / (count - 1)) ** 2
    phase = 2 * np.pi * (high * t + (low - high) * t * t / (2 * MARKER_SECONDS))
    return (MARKER_AMPLITUDE * envelope * np.sin(phase)).astype(np.float32)


def query(*command):
    return subprocess.check_output(command, text=True, timeout=5).strip()


def service_identity():
    fields = ("ActiveState", "SubState", "MainPID", "NRestarts", "ExecMainStartTimestamp")
    values = query("systemctl", "show", "vc1.service",
                   *(f"--property={field}" for field in fields))
    return dict(line.split("=", 1) for line in values.splitlines())


def source_full_scale_thresholds(source):
    if source.get("mute") is not False:
        raise RuntimeError("Input source is muted; refusing an invalid full-scale check")
    volume = source.get("volume")
    if not isinstance(volume, dict):
        raise RuntimeError("Input source volume is unavailable; refusing an invalid full-scale check")
    thresholds = []
    for channel in SOURCE_CHANNELS:
        channel_volume = volume.get(channel)
        value = channel_volume.get("value") if isinstance(channel_volume, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"Input source volume for {channel} is invalid")
        # PulseAudio/PipeWire volume values use cubic amplitude scaling.
        thresholds.append(0.999 * (value / 65536) ** 3)
    return tuple(thresholds)


def check_route(output, output_port="analog-output-headphones-es8389-m2"):
    if query("pactl", "get-default-source") != SOURCE:
        raise RuntimeError("Default source changed; refusing fallback capture")
    sources = json.loads(query("pactl", "-f", "json", "list", "sources"))
    source = next((s for s in sources if s["name"] == SOURCE), None)
    if source is None or source.get("active_port") != "analog-input-headset-mic-es8389-m2":
        raise RuntimeError("Expected headset microphone is not selected")
    input_full_scale_thresholds = source_full_scale_thresholds(source)
    sink = None
    if output:
        if query("pactl", "get-default-sink") != SINK:
            raise RuntimeError("Default sink changed; refusing fallback output")
        sinks = json.loads(query("pactl", "-f", "json", "list", "sinks"))
        sink = next((s for s in sinks if s["name"] == SINK), None)
        if sink is None or sink.get("active_port") != output_port:
            raise RuntimeError("Explicitly confirmed output port is not selected")
        if sink.get("mute"):
            raise RuntimeError("Output muted; adjust manually before confirming safety")
    return {"source": SOURCE, "source_volume": source["volume"],
            "input_full_scale_thresholds": input_full_scale_thresholds,
            "sink": SINK if output else None,
            "output_port": output_port if output else None,
            "sink_volume": sink.get("volume") if sink is not None else None}


def ensure_source_volume_unchanged(expected_route, current_route):
    if current_route["source_volume"] != expected_route["source_volume"]:
        raise RuntimeError("Input source volume changed during probe")


def graph_snapshot():
    objects = json.loads(query("pw-dump"))
    nodes = []
    for obj in objects:
        if obj["type"] != "PipeWire:Interface:Node":
            continue
        props = obj.get("info", {}).get("props", {})
        if "Audio" in props.get("media.class", ""):
            nodes.append({"id": obj["id"], "state": obj["info"].get("state"),
                          "props": {k: props[k] for k in (
                              "node.name", "media.class", "application.name",
                              "audio.rate", "audio.channels", "node.latency")
                              if k in props}})
    ids = {n["id"] for n in nodes}
    links = []
    for obj in objects:
        if obj["type"] == "PipeWire:Interface:Link":
            info = obj["info"]
            if info.get("output-node-id") in ids or info.get("input-node-id") in ids:
                links.append({k: info[k] for k in ("output-node-id", "input-node-id", "state")
                              if k in info})
    pcm = {}
    for direction in ("c", "p"):
        for name in ("status", "hw_params"):
            path = Path(f"/proc/asound/card0/pcm0{direction}/sub0/{name}")
            pcm[str(path)] = path.read_text()
    return {"nodes": nodes, "links": links, "pcm": pcm,
            "clock_metadata": query("pw-metadata", "-n", "settings")}


class Samples:
    """Fixed-size storage: no audio retained and no I/O in callbacks."""

    def __init__(self, mode, rate, blocksize, seconds, markers=False,
                 input_full_scale_thresholds=(0.999, 0.999)):
        self.mode = mode
        self.blocksize = blocksize
        if len(input_full_scale_thresholds) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
                for value in input_full_scale_thresholds):
            raise ValueError("Expected two finite, positive input full-scale thresholds")
        self.input_full_scale_thresholds = np.asarray(input_full_scale_thresholds, dtype=np.float32)
        self.total = round(rate * seconds)
        self.position = 0
        self.callbacks = 0
        self.input_peak = 0.0
        self.input_near_full_scale_samples = 0
        self.output_peak = 0.0
        self.fault = None
        self.ramp = np.arange(blocksize, dtype=np.float32)
        self.gain = np.empty(blocksize, dtype=np.float32)
        self.scratch = np.empty((blocksize, 2), dtype=np.float32)
        self.fade = round(rate * 0.02)
        self.tone = None
        self.marker = None
        self.marker_starts = ()
        if markers:
            wave = marker_waveform(rate)
            self.marker = np.repeat(wave[:, None], 2, axis=1)
            self.marker_starts = tuple(
                [round(s * rate) for s in MARKER_START_SECONDS]
                + [self.total - round(s * rate) for s in MARKER_END_SECONDS_BEFORE_STOP])
        if mode == "tone":
            t = np.arange(self.total, dtype=np.float32) / rate
            self.tone = (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def marker_report(self):
        if self.marker is None:
            return None
        starts = list(self.marker_starts)
        return {
            "waveform": "Hann^2 linear sweep, identical on both channels, added after fades",
            "sweep_hz": list(MARKER_SWEEP_HZ), "seconds": MARKER_SECONDS,
            "amplitude": MARKER_AMPLITUDE,
            "start_marker_samples": starts[:len(MARKER_START_SECONDS)],
            "end_marker_samples": starts[len(MARKER_START_SECONDS):],
            "full_gain_output_samples": [self.fade, self.total - 1 - self.fade],
            "total_output_samples": self.total,
            "mapping": "output sample p reaches the recording at t_start_marker + "
                       "(p - start_marker_samples[0]) / samplerate (plus clock drift)",
        }

    def process(self, data, out, frames):
        if frames != self.blocksize:
            self.fault = "Unexpected callback frame count"
            return False
        if data is not None:
            if not np.isfinite(data).all():
                self.fault = "Non-finite input"
                return False
            np.abs(data, out=self.scratch)
            self.input_peak = max(self.input_peak, float(self.scratch.max()))
            self.input_near_full_scale_samples += int(np.count_nonzero(
                self.scratch >= self.input_full_scale_thresholds))
        if out is not None:
            if self.mode == "tone":
                end = min(self.position + frames, self.total)
                count = max(0, end - self.position)
                out.fill(0)
                out[:count, 0] = self.tone[self.position:end]
                out[:count, 1] = out[:count, 0]
            else:
                # Stereo is preserved; conservative gain, then a final ceiling.
                np.multiply(data, 0.1, out=out)
            np.add(self.ramp, self.position, out=self.gain)
            np.minimum(self.gain, self.total - 1 - self.gain, out=self.gain)
            np.divide(self.gain, self.fade, out=self.gain)
            np.clip(self.gain, 0, 1, out=self.gain)
            np.multiply(out, self.gain[:, None], out=out)
            if self.marker is not None:
                length = len(self.marker)
                for start in self.marker_starts:
                    lo = max(start, self.position)
                    hi = min(start + length, self.position + frames)
                    if lo < hi:
                        target = out[lo - self.position:hi - self.position]
                        np.add(target, self.marker[lo - start:hi - start], out=target)
            np.clip(out, -0.1, 0.1, out=out)
            np.abs(out, out=self.scratch)
            self.output_peak = max(self.output_peak, float(self.scratch.max()))
        self.position += frames
        self.callbacks += 1
        return self.position < self.total


def run(args, sd, report):
    output = args.mode != "capture"
    if output and not args.safety_confirmed:
        raise ValueError("Output requires --safety-confirmed after onsite headphone/volume confirmation")
    if args.markers and args.mode != "through":
        raise ValueError("--markers is only valid with --mode through")
    report["route"] = check_route(output, args.output_port)
    report["input_full_scale_thresholds"] = list(report["route"]["input_full_scale_thresholds"])
    report["input_full_scale_threshold_model"] = (
        "0.999 times per-channel cubic PulseAudio/PipeWire software source gain")
    before = service_identity()
    report["service_before"] = before
    if before.get("ActiveState") != "active" or before.get("SubState") != "running":
        raise RuntimeError("RVC service state changed; stop and investigate separately")
    devices = sd.query_devices()
    matches = [d for d in devices if d["name"] == args.device
               and sd.query_hostapis(d["hostapi"])["name"] == "ALSA"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one ALSA pipewire device; refusing fallback")
    device = matches[0]
    device_index = device["index"]
    host = sd.query_hostapis(device["hostapi"])["name"]
    if device["name"] != "pipewire" or host != "ALSA":
        raise ValueError("This probe permits only the explicitly selected ALSA pipewire device")
    report["device"] = dict(device)
    report["hostapi"] = host
    report["portaudio"] = sd.get_portaudio_version()
    if args.mode != "tone":
        sd.check_input_settings(device=device_index, channels=2, dtype="float32", samplerate=48000)
    if output:
        sd.check_output_settings(device=device_index, channels=2, dtype="float32", samplerate=48000)
    samples = Samples(args.mode, 48000, args.blocksize, args.seconds, args.markers,
                      report["route"]["input_full_scale_thresholds"])
    report["markers"] = samples.marker_report()

    def transfer(data, out, frames, status):
        if out is not None:
            out.fill(0)
        if status:
            samples.fault = "PortAudio callback reported an xrun or other status"
            raise sd.CallbackAbort
        if not samples.process(data, out, frames):
            if samples.fault:
                if out is not None:
                    out.fill(0)
                raise sd.CallbackAbort
            raise sd.CallbackStop

    options = dict(device=device_index, samplerate=48000, blocksize=args.blocksize,
                   channels=2, dtype="float32", latency="low")
    if args.mode == "capture":
        stream = sd.InputStream(**options, callback=lambda d, n, t, s: transfer(d, None, n, s))
    elif args.mode == "tone":
        stream = sd.OutputStream(**options, callback=lambda o, n, t, s: transfer(None, o, n, s))
    else:
        stream = sd.Stream(**options, callback=lambda d, o, n, t, s: transfer(d, o, n, s))
    cpu_samples = []
    started = time.monotonic()
    cpu_start = time.process_time()
    previous_wall, previous_cpu = started, cpu_start
    try:
        with stream:
            latency = stream.latency
            report["reported_latency_seconds_NOT_end_to_end"] = latency
            report["actual_samplerate"] = stream.samplerate
            report["actual_channels"] = stream.channels
            report["active_graph"] = graph_snapshot()
            while stream.active:
                time.sleep(0.25)
                current_route = check_route(output, args.output_port)
                ensure_source_volume_unchanged(report["route"], current_route)
                if service_identity() != before:
                    raise RuntimeError("RVC identity/state changed during probe")
                now, cpu = time.monotonic(), time.process_time()
                cpu_samples.append(100 * (cpu - previous_cpu) / (now - previous_wall))
                previous_wall, previous_cpu = now, cpu
                if now - started > args.seconds + 10:
                    raise RuntimeError("Probe exceeded deadline")
            if samples.fault:
                raise RuntimeError(samples.fault)
            if samples.position < samples.total:
                raise RuntimeError("Stream stopped before requested duration")
    finally:
        stream.close()
        report.update(
            callbacks=samples.callbacks, frames_processed=samples.position,
            input_peak=samples.input_peak, output_peak=samples.output_peak,
            input_near_full_scale_samples=samples.input_near_full_scale_samples,
            input_quality=("suspect_full_scale" if samples.input_near_full_scale_samples
                           else "no_full_scale_observed"),
            callback_fault=samples.fault, wall_seconds=time.monotonic() - started,
            process_cpu_seconds=time.process_time() - cpu_start,
            process_cpu_raw_percent_samples=cpu_samples,
            online_cpu_count=os.cpu_count(), stream_closed=stream.closed,
            service_after=service_identity(),
        )
    if report["service_after"] != before:
        raise RuntimeError("RVC identity/state changed at exit")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("list", "capture", "tone", "through"), default="list")
    p.add_argument("--device", choices=("pipewire",))
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--blocksize", type=int, choices=(256, 512, 1024), default=1024)
    p.add_argument("--safety-confirmed", action="store_true")
    p.add_argument("--markers", action="store_true",
                   help="through only: add low-level acoustic on/off sweeps for recording alignment")
    p.add_argument("--output-port", choices=("analog-output-headphones-es8389-m2",
                                           "analog-output-es8389-m2"),
                   default="analog-output-headphones-es8389-m2")
    p.add_argument("--report", type=Path)
    return p


def main():
    args = parser().parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 60:
        raise SystemExit("--seconds must be finite and between 1 and 60")
    import sounddevice as sd
    if args.mode == "list":
        print(sd.query_devices())
        print(sd.query_hostapis())
        return 0
    if args.device is None or args.report is None:
        raise SystemExit("Active probes require --device and --report")
    report = {"mode": args.mode, "samplerate": 48000, "channels": 2,
              "requested_blocksize": args.blocksize, "seconds": args.seconds,
              "end_to_end_latency": "UNVERIFIED: no physical reference measurement"}
    # Reserve the evidence file before opening any stream; never overwrite a run.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as evidence:
        try:
            run(args, sd, report)
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError, sd.PortAudioError) as error:
            report.update(outcome="failed", error=str(error))
            print(f"FAILED: {error}", file=sys.stderr)
        except KeyboardInterrupt:
            report.update(outcome="interrupted", error="Stopped by operator")
            print("INTERRUPTED: stopped by operator", file=sys.stderr)
        else:
            report["outcome"] = "completed"
        json.dump(report, evidence, indent=2)
        evidence.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["outcome"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
