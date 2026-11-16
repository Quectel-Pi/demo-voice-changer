"""Real DSP/storage with synthetic device callbacks; hardware is never opened."""

from ctypes.util import find_library
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

import numpy as np

from voice_changer.audio.analysis import AudioAnalysis, SignalAnalyzer
from voice_changer.audio.device_types import CaptureConfig, DeviceRef, OutputConfig
from voice_changer.audio.engine import AudioEngine, EngineError
from voice_changer.audio.media_backend import CaptureBackend, PlaybackBackend
from voice_changer.audio.recording import RecordingWriter
from voice_changer.audio.sounddevice_backend import SoundDeviceBackend
from voice_changer.dsp.parameters import LiveControl, PRESETS

from media_fakes import MediaLibrary


INPUT = DeviceRef(0, "test input", 0)
OUTPUT = DeviceRef(1, "test output", 0)
CAPTURE = CaptureConfig(INPUT, 48000, 512, 1, 0.05)
PLAYBACK = OutputConfig(OUTPUT, 48000, 512, 1, 0.05)
NATIVE = os.environ.get("VOICE_CHANGER_RUBBERBAND_LIBRARY") or find_library("rubberband")


class MediaBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library = MediaLibrary()
        self.patcher = patch.dict(sys.modules, {"sounddevice": self.library})
        self.patcher.start()
        self.engine = None
        self.writer = None
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for raw in self.library.streams:
            raw.close_error = None
        if self.engine is not None:
            self.engine.stop()
        if self.writer is not None:
            self.writer.finish(complete=False, reason="test cleanup")
        self.patcher.stop()
        self.temp.cleanup()
        self.assertFalse(any(
            t.name in (
                "voice-changer-dsp", "voice-changer-analysis",
                "synthetic-media-device", "recording-disk-writer",
            )
            for t in threading.enumerate()
        ))

    def capture(self, config=CAPTURE):
        backend = CaptureBackend(config, LiveControl(PRESETS["original"]))
        self.engine = AudioEngine(backend)
        self.engine.start()
        self.assertIsNotNone(backend.stream)
        return backend.stream

    def test_equal_input_callbacks_publish_distinct_capture_snapshots(self):
        class FakeProcessor:
            error = None

            def exchange(self, samples, output):
                output.fill(0)
                return True

            def close(self):
                pass

        control = LiveControl(PRESETS["original"])
        stream = CaptureBackend(CAPTURE, control).open_stream()
        stream.processor = FakeProcessor()
        self.addCleanup(stream.close)
        samples = np.zeros((CAPTURE.blocksize, CAPTURE.channels), dtype=np.float32)
        stream._capture(samples, CAPTURE.blocksize, None, False)
        first = control.levels
        stream._capture(samples, CAPTURE.blocksize, None, False)
        self.assertEqual(first, (0.0, 0.0))
        self.assertEqual(control.levels, first)
        self.assertIsNot(control.levels, first)

    def test_analysis_overflow_does_not_abort_or_block_capture_callback(self):
        class FakeProcessor:
            error = None

            def exchange(self, samples, output):
                output[:] = samples
                return True

            def close(self):
                pass

        control = LiveControl(PRESETS["original"])
        stream = CaptureBackend(CAPTURE, control).open_stream()
        stream.processor = FakeProcessor()
        stream.analysis = AudioAnalysis(48000, 512, 1, capture=True)
        control.analysis = stream.analysis
        stream.analysis._accepting = True
        self.addCleanup(stream.close)
        samples = np.full((512, 1), 0.02, dtype=np.float32)
        for _ in range(stream.analysis.capacity // 512 + 1):
            stream._capture(samples, 512, None, False)
        self.assertIsNone(stream._fault)
        self.assertGreater(stream.analysis._dropped, 0)
        self.assertAlmostEqual(control.levels[0], 0.02)
        stream.analysis.start()
        deadline = time.monotonic() + 2
        while stream.analysis.snapshot is None and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertGreater(stream.analysis.snapshot.dropped_blocks, 0)

    def test_analysis_buffer_failure_is_reported_without_aborting_capture(self):
        class FakeProcessor:
            error = None

            def exchange(self, samples, output):
                output[:] = samples
                return True

            def close(self):
                pass

        control = LiveControl(PRESETS["original"])
        stream = CaptureBackend(CAPTURE, control).open_stream()
        stream.processor = FakeProcessor()
        stream.analysis = AudioAnalysis(48000, 512, 1, capture=True)
        control.analysis = stream.analysis
        stream.analysis._data.setflags(write=False)
        stream.analysis.start()
        self.addCleanup(stream.close)
        samples = np.full((512, 1), 0.02, dtype=np.float32)
        stream._capture(samples, 512, None, False)
        self.assertIsNone(stream._fault)
        self.assertAlmostEqual(control.levels[0], 0.02)
        deadline = time.monotonic() + 2
        while stream.analysis.snapshot is None and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertIn("read-only", stream.analysis.snapshot.input.error)
        self.assertIn("read-only", stream.analysis.snapshot.output.error)

    def test_analysis_close_failure_retains_worker_for_retry_in_all_streams(self):
        from voice_changer.audio.device_types import StreamConfig

        config = StreamConfig(INPUT, OUTPUT, 48000, 512, 1, 0.05)
        for kind in ("capture", "duplex", "playback"):
            with self.subTest(kind=kind):
                control = LiveControl(PRESETS["original"])
                if kind == "capture":
                    stream = CaptureBackend(CAPTURE, control).open_stream()
                    attribute = "analysis"
                elif kind == "duplex":
                    stream = SoundDeviceBackend(
                        config, output_confirmed=True, control=control,
                    ).open_stream()
                    attribute = "_analysis"
                else:
                    stream = PlaybackBackend(
                        PLAYBACK, self.root / "not-opened.wav", 512, output_confirmed=True,
                    ).open_stream()
                    attribute = "analysis"
                worker = AudioAnalysis(48000, 512, 1, capture=kind != "playback")
                setattr(stream, attribute, worker)
                if kind != "playback":
                    control.analysis = worker
                with patch.object(
                    worker, "close", side_effect=(RuntimeError("analysis worker stuck"), None),
                ):
                    error_type = EngineError if kind == "capture" else RuntimeError
                    with self.assertRaisesRegex(error_type, "analysis worker stuck"):
                        stream.close()
                    self.assertIs(getattr(stream, attribute), worker)
                    self.assertFalse(stream._closed)
                    if kind != "playback":
                        self.assertIs(control.analysis, worker)
                    stream.close()
                self.assertIsNone(getattr(stream, attribute))
                self.assertTrue(stream._closed)
                if kind != "playback":
                    self.assertIsNone(control.analysis)

    def record(self, stream):
        self.writer = RecordingWriter(48000, stream.config.channels, 512, data_root=self.root)
        stream.start_recording(self.writer)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; live analysis is not verified")
    def test_capture_and_duplex_analyze_real_callback_buffers_and_release_worker(self):
        for duplex in (False, True):
            with self.subTest(duplex=duplex):
                control = LiveControl(PRESETS["original"])
                if duplex:
                    from voice_changer.audio.device_types import StreamConfig
                    config = StreamConfig(INPUT, OUTPUT, 48000, 512, 1, 0.05)
                    backend = SoundDeviceBackend(config, output_confirmed=True, control=control)
                else:
                    backend = CaptureBackend(CAPTURE, control)
                self.engine = AudioEngine(backend)
                self.engine.start()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    snapshot = control.analysis.snapshot
                    if snapshot is not None and snapshot.output and snapshot.output.pitch_hz:
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(snapshot)
                self.assertAlmostEqual(snapshot.input.pitch_hz, 220, delta=1)
                self.assertAlmostEqual(snapshot.output.pitch_hz, 220, delta=1)
                self.assertGreater(snapshot.input.bands[0], 0)
                self.assertGreater(snapshot.output.bands[0], 0)
                self.engine.stop()
                self.assertIsNone(control.analysis)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; live analysis is not verified")
    def test_analysis_exception_is_visible_without_stopping_audio(self):
        with patch.object(SignalAnalyzer, "analyze", side_effect=ValueError("synthetic analysis failure")):
            stream = self.capture()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                snapshot = stream.analysis.snapshot
                if snapshot is not None and snapshot.input and snapshot.input.error:
                    break
                time.sleep(0.01)
            self.assertIn("synthetic analysis failure", snapshot.input.error)
            self.assertIn("synthetic analysis failure", snapshot.output.error)
            self.assertIsNone(stream._fault)
            self.engine.check_health()
            self.assertTrue(self.library.streams[0].active)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; capture DSP is not verified")
    def test_input_only_record_monitor_toggle_and_paired_original(self):
        stream = self.capture()
        self.assertEqual([raw.kind for raw in self.library.streams], ["input"])
        self.assertFalse(any(kind == "output" for op, kind, ident in self.library.operations))
        self.record(stream)
        time.sleep(0.08)
        stream.enable_monitor(PLAYBACK, output_confirmed=True)
        time.sleep(0.1)
        stream.disable_monitor()
        time.sleep(0.08)
        self.engine.check_health()
        group = stream.stop_recording()
        self.assertTrue(group.complete)
        self.assertGreater(group.frames, 2048)
        self.assertTrue(self.library.streams[0].active)
        self.assertTrue(self.library.streams[1].closed)
        with wave.open(str(group.original_path), "rb") as original, wave.open(str(group.effect_path), "rb") as effect:
            self.assertEqual(original.getnframes(), effect.getnframes())
            raw_pcm = original.readframes(group.frames)
            effect_pcm = effect.readframes(group.frames)
        metadata = json.loads(group.metadata_path.read_text())
        # The spectral-gate denoiser runs on every block (even at 0%) so its
        # fixed lookahead keeps the recording pair explainable; the original
        # track stays pre-gain and untouched, the effect track is the raw
        # capture shifted by the documented denoise lookahead only.
        hop = metadata["alignment"]["denoise_delay_frames"]
        self.assertEqual(hop, 256)
        raw_frames = np.frombuffer(raw_pcm, dtype="<i2")
        effect_frames = np.frombuffer(effect_pcm, dtype="<i2")
        np.testing.assert_array_equal(effect_frames[hop:], raw_frames[: group.frames - hop])
        self.assertIn("80Hz", metadata["alignment"]["effect_algorithms"])
        self.assertIn("0.8s", metadata["alignment"]["effect_algorithms"])
        self.assertIn("original track pre-gain", metadata["alignment"]["gain_stages"])
        self.assertEqual(metadata["source_interval"]["end_exclusive"] - metadata["source_interval"]["start"], group.frames)
        positions = np.arange(group.frames) + metadata["source_interval"]["start"]
        expected = 0.02 * np.sin(2 * np.pi * 220 * positions / 48000)
        actual = np.frombuffer(raw_pcm, dtype="<i2").astype(np.float32) / 32768
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1 / 32768)
        self.assertEqual(metadata["alignment"]["native_start_pad_frames"], stream.processor._dsp.start_pad)
        self.assertEqual(metadata["alignment"]["native_trimmed_start_delay_frames"], stream.processor._dsp.start_delay)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; capture DSP is not verified")
    def test_monitor_open_failure_does_not_stop_input_only_recording(self):
        stream = self.capture()
        self.record(stream)
        self.library.output_error = "synthetic output unavailable"
        with self.assertRaisesRegex(Exception, "output unavailable"):
            stream.enable_monitor(PLAYBACK, output_confirmed=True)
        self.assertFalse(stream.monitoring)
        time.sleep(0.1)
        self.engine.check_health()
        self.assertTrue(stream.stop_recording().complete)

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; capture DSP is not verified")
    def test_recording_preserves_pre_limit_raw_and_limits_effect_in_mono_and_stereo(self):
        self.library.amplitude = 0.2
        self.library.devices[0]["max_input_channels"] = 2
        for channels in (1, 2):
            with self.subTest(channels=channels):
                stream = self.capture(CaptureConfig(INPUT, 48000, 512, channels, 0.05))
                self.record(stream)
                time.sleep(0.1)
                group = stream.stop_recording()
                with wave.open(str(group.original_path), "rb") as original, wave.open(str(group.effect_path), "rb") as effect:
                    self.assertEqual(original.getnchannels(), channels)
                    raw = np.frombuffer(original.readframes(group.frames), dtype="<i2")
                    limited = np.frombuffer(effect.readframes(group.frames), dtype="<i2")
                self.assertGreater(float(np.abs(raw).max()) / 32768, 0.15)
                self.assertLessEqual(float(np.abs(limited).max()) / 32768, 0.05 + 1 / 32768)
                self.engine.stop()

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; gain/DSP capture is not verified")
    def test_nonunity_gains_and_fun_effects_preserve_raw_and_limit_stereo_recording(self):
        self.library.devices[0]["max_input_channels"] = 2
        parameters = replace(PRESETS["monster"], input_gain_db=12, output_gain_db=12, reverb_mix=0.2)
        for channels in (1, 2):
            with self.subTest(channels=channels):
                control = LiveControl(parameters)
                backend = CaptureBackend(replace(CAPTURE, channels=channels), control)
                self.engine = AudioEngine(backend)
                self.engine.start()
                stream = backend.stream
                self.record(stream)
                time.sleep(0.35)
                self.engine.check_health()
                group = stream.stop_recording()
                self.assertTrue(group.complete)
                metadata = json.loads(group.metadata_path.read_text())
                with wave.open(str(group.original_path), "rb") as original, wave.open(str(group.effect_path), "rb") as effect:
                    raw = np.frombuffer(original.readframes(group.frames), dtype="<i2").astype(np.float32).reshape(-1, channels) / 32768
                    wet = np.frombuffer(effect.readframes(group.frames), dtype="<i2").astype(np.float32).reshape(-1, channels) / 32768
                positions = np.arange(group.frames) + metadata["source_interval"]["start"]
                expected = np.repeat(
                    (0.02 * np.sin(2 * np.pi * 220 * positions / 48000))[:, None], channels, axis=1,
                )
                np.testing.assert_allclose(raw, expected, rtol=0, atol=1 / 32768)
                self.assertLessEqual(float(np.abs(wet).max()), 0.05)
                self.assertFalse(np.array_equal(raw, wet))
                events = [json.loads(line) for line in (group.directory / "parameters.jsonl").read_text().splitlines()]
                self.assertEqual(events[0]["parameters"]["input_gain_db"], 12)
                self.assertEqual(events[0]["parameters"]["output_gain_db"], 12)
                self.assertEqual(events[0]["parameters"]["reverb_mix"], 0.2)
                self.engine.stop()

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; capture DSP is not verified")
    def test_runtime_dsp_fault_finalizes_incomplete_and_releases(self):
        stream = self.capture()
        self.record(stream)
        time.sleep(0.1)
        stream.processor.error = RuntimeError("synthetic DSP failure")
        with self.assertRaisesRegex(EngineError, "synthetic DSP failure"):
            self.engine.check_health()
        self.assertFalse(self.engine.has_stream)
        self.assertFalse(self.writer.group.complete)
        self.assertTrue(all(raw.closed for raw in self.library.streams))

    @unittest.skipUnless(NATIVE, "Rubber Band unavailable; capture DSP is not verified")
    def test_input_close_failure_keeps_dsp_until_retry(self):
        stream = self.capture()
        raw = self.library.streams[0]
        raw.close_error = RuntimeError("synthetic close failure")
        with self.assertRaisesRegex(EngineError, "synthetic close failure"):
            self.engine.stop()
        self.assertTrue(self.engine.has_stream)
        self.assertTrue(stream.processor._thread.is_alive())
        raw.close_error = None
        self.engine.stop()
        self.assertFalse(self.engine.has_stream)
        self.assertIsNone(stream.processor)

    def wav(self, frames):
        path = self.root / "synthetic.wav"
        samples = (np.arange(frames, dtype=np.int64) % 60000 - 30000).astype("<i2")
        with wave.open(str(path), "wb") as target:
            target.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            target.writeframes(samples.tobytes())
        return path, samples

    def playback(self, frames):
        self.library.auto_callbacks = False
        path, samples = self.wav(frames)
        backend = PlaybackBackend(PLAYBACK, path, frames, output_confirmed=True)
        self.engine = AudioEngine(backend)
        self.engine.start()
        return backend.stream, samples

    def test_playback_uses_only_output_and_callback_never_reads_file(self):
        stream, samples = self.playback(100)
        self.assertEqual([raw.kind for raw in self.library.streams], ["output"])
        self.assertFalse(any(kind == "input" for op, kind, ident in self.library.operations))
        with patch.object(stream._reader, "readframes", side_effect=AssertionError("callback file I/O")) as reader:
            output = self.library.streams[0].tick()
        reader.assert_not_called()
        self.assertTrue(stream.finished)
        self.assertLessEqual(float(np.abs(output).max()), 0.05)
        np.testing.assert_array_equal(output[100:], 0)
        self.engine.check_health()

    def test_playback_peak_is_unknown_until_a_silent_callback(self):
        self.library.auto_callbacks = False
        path, _ = self.wav(1024)
        with wave.open(str(path), "wb") as target:
            target.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            target.writeframes(bytes(2048))
        backend = PlaybackBackend(PLAYBACK, path, 1024, output_confirmed=True)
        self.engine = AudioEngine(backend)
        self.engine.start()
        self.assertIsNone(backend.stream.output_peak)
        self.assertIsNone(backend.stream.output_peak_sample)
        self.library.streams[0].tick()
        self.assertEqual(backend.stream.output_peak, 0.0)
        self.assertEqual(backend.stream.output_peak_sample, (1, 0.0))
        self.library.streams[0].tick()
        self.assertEqual(backend.stream.output_peak_sample, (2, 0.0))

    def test_playback_analysis_follows_output_callback_not_prefetched_pcm(self):
        self.library.auto_callbacks = False
        frames = 512 * 24
        phase = np.arange(frames)
        samples = np.concatenate((
            0.1 * np.sin(2 * np.pi * 100 * phase[:512 * 12] / 48000),
            0.1 * np.sin(2 * np.pi * 4000 * phase[512 * 12:] / 48000),
        ))
        path = self.root / "synthetic-tones.wav"
        with wave.open(str(path), "wb") as target:
            target.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            target.writeframes((samples * 32767).astype("<i2").tobytes())
        backend = PlaybackBackend(PLAYBACK, path, frames, output_confirmed=True)
        self.engine = AudioEngine(backend)
        self.engine.start()
        stream = backend.stream
        self.assertEqual(stream._remaining, 0)
        self.assertIsNone(stream.analysis.snapshot)
        for _ in range(10):
            output = self.library.streams[0].tick()
        deadline = time.monotonic() + 2
        while stream.analysis.snapshot is None and time.monotonic() < deadline:
            time.sleep(0.005)
        snapshot = stream.analysis.snapshot
        self.assertIsNotNone(snapshot)
        self.assertIsNone(snapshot.input)
        self.assertAlmostEqual(snapshot.output.pitch_hz, 100, delta=1)
        self.assertGreater(snapshot.output.bands[0], snapshot.output.bands[2])
        self.assertLessEqual(float(np.abs(output).max()), 0.05)
        self.engine.stop()
        self.assertIsNone(stream.analysis)

    def test_playback_prefetch_preserves_pcm_without_effects_or_normalization(self):
        stream, samples = self.playback(512 * 65 + 7)
        blocks = []
        while not stream.finished:
            blocks.append(self.library.streams[0].tick().copy())
            stream.fill()
        actual = np.concatenate(blocks)[:len(samples), 0]
        expected = np.clip(samples.astype(np.float32) / 32768, -stream._ceiling, stream._ceiling)
        np.testing.assert_array_equal(actual, expected)

    def test_playback_starvation_reports_fault_and_closes(self):
        stream, _ = self.playback(512 * 65)
        for _ in range(65):
            self.library.streams[0].tick()
        with self.assertRaisesRegex(EngineError, "underrun"):
            self.engine.check_health()
        self.assertFalse(self.engine.has_stream)
        self.assertIsNone(stream._reader)

    def test_playback_malformed_format_and_unconfirmed_output_fail_closed(self):
        path, _ = self.wav(512)
        backend = PlaybackBackend(PLAYBACK, path, 512)
        with self.assertRaisesRegex(RuntimeError, "Confirm safe output"):
            backend.open_stream()
        with wave.open(str(path), "wb") as target:
            target.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            target.writeframes(bytes(1024))
        self.engine = AudioEngine(PlaybackBackend(PLAYBACK, path, 512, output_confirmed=True))
        with self.assertRaisesRegex(EngineError, "format"):
            self.engine.start()
        self.assertFalse(self.engine.has_stream)

    def test_playback_rejects_changed_track_length(self):
        self.library.auto_callbacks = False
        path, _ = self.wav(513)
        self.engine = AudioEngine(PlaybackBackend(PLAYBACK, path, 512, output_confirmed=True))
        with self.assertRaisesRegex(EngineError, "format/interval"):
            self.engine.start()
        self.assertFalse(self.engine.has_stream)
