"""Qt envelope tests use synthetic peak values and never open audio devices."""

import math
import os
import time
import unittest
from contextlib import ExitStack
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QLabel, QStyle, QStyleOption

from voice_changer.audio.analysis_status import AnalysisReading
from voice_changer.ui.audio_wave import (
    ANALYSIS_INVALID, AudioWave, INVALID, MAX_PITCH_POINTS, MAX_POINTS,
    NO_CAPTURE, WAITING, WINDOW_SECONDS, WaveComparison,
    _height_for_peak,
)
from voice_changer.ui.theme import COLORS, set_visual_state, stylesheet

from qt_wait import wait_until


class AudioWaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.meter = AudioWave("麦克风 / Microphone")
        self.addCleanup(self.meter.close)

    def test_silence_unknown_invalid_and_full_scale_are_distinct(self):
        self.assertEqual(self.meter.display_text, NO_CAPTURE)
        self.assertIsNone(self.meter.peak)
        self.assertEqual(self.meter.history, ())
        self.meter.set_waiting()
        self.assertEqual(self.meter.display_text, WAITING)
        self.assertIsNone(self.meter.peak)
        self.meter.set_peak(None)
        self.assertEqual(self.meter.display_text, WAITING)
        self.assertIsNone(self.meter.peak)
        for peak, text, height in (
            (0.0, "−∞ dBFS", 0.0),
            (0.01, "-40.0 dBFS", 1 / 3),
            (0.1, "-20.0 dBFS", 2 / 3),
            (1.0, "0.0 dBFS", 1.0),
            (2.0, "6.0 dBFS", 1.0),
        ):
            with self.subTest(peak=peak):
                self.meter.set_peak(peak)
                self.assertEqual(self.meter.display_text, text)
                self.assertEqual(self.meter.peak, peak)
                self.assertAlmostEqual(_height_for_peak(peak), height)
                self.assertIn(text, self.meter.accessibleDescription())
        self.assertEqual(_height_for_peak(0.0001), 0.0)
        for invalid in (-1.0, math.nan, math.inf, -math.inf, True):
            with self.subTest(invalid=invalid):
                self.meter.set_peak(invalid)
                self.assertEqual(self.meter.display_text, INVALID)
                self.assertIsNone(self.meter.peak)
                self.assertEqual(self.meter.history, ())
                self.assertFalse(self.meter._timer.isActive())
                self.meter.set_analysis(AnalysisReading((0.1, 0.1, 0.1), 220))
                self.meter.set_analysis_waiting()
                self.assertIn(ANALYSIS_INVALID, self.meter.analysis_status.text())
                self.assertIsNone(self.meter.analysis_canvas.reading)
        self.meter.set_idle()
        self.assertEqual(self.meter.display_text, NO_CAPTURE)

    def test_live_readings_do_not_move_headers_or_wave_layout(self):
        second = AudioWave("处理后 / Processed")
        comparison = WaveComparison(self.meter, second)
        self.addCleanup(comparison.close)
        widgets = (
            self.meter, second, self.meter.name_label, second.name_label,
            self.meter.reading_label, second.reading_label,
            self.meter.canvas, second.canvas,
        )
        comparison.show()
        for width, side in ((440, True), (900, False), (900, True)):
            with self.subTest(width=width, side_by_side=side):
                comparison.resize(width, 420)
                comparison.set_side_by_side(side)
                self.meter.set_peak(0.1)
                second.set_peak(0.1)
                self.app.processEvents()
                self.app.processEvents()
                baseline = tuple(widget.geometry() for widget in widgets)
                for index in range(80):
                    peak = (0.0, 0.00001, 0.1, 1.0, 5e-324, 1e308)[index % 6]
                    self.meter.set_peak(peak)
                    second.set_peak(peak)
                    self.app.processEvents()
                    self.assertEqual(tuple(widget.geometry() for widget in widgets), baseline)
                    for meter in (self.meter, second):
                        self.assertGreaterEqual(
                            meter.reading_label.height(),
                            meter.reading_label.heightForWidth(meter.reading_label.width()),
                        )

    def test_history_is_time_ordered_bounded_and_cleared_on_stop(self):
        times = [index * 0.05 for index in range(100)]
        with patch("voice_changer.ui.audio_wave.monotonic", side_effect=times):
            for index in range(100):
                self.meter.set_peak(index / 100)
                self.meter._sample()
        history = self.meter.history
        self.assertEqual(len(history), MAX_POINTS)
        self.assertEqual(tuple(timestamp for timestamp, _ in history), tuple(sorted(t for t, _ in history)))
        self.assertGreaterEqual(history[0][0], history[-1][0] - WINDOW_SECONDS)
        self.assertAlmostEqual(history[-1][1], 0.99)
        self.meter._sample()
        self.assertEqual(self.meter.history, history)
        self.meter.set_idle()
        self.assertEqual(self.meter.history, ())
        self.assertIsNone(self.meter.peak)
        with patch("voice_changer.ui.audio_wave.monotonic", side_effect=(0.0, 0.2, 3.0, 3.21)):
            for peak in (0.1, 0.2, 0.3, 0.4):
                self.meter.set_peak(peak)
                self.meter._sample()
        self.assertEqual(self.meter.history, ((3.0, 0.3), (3.21, 0.4)))

    def test_rendered_envelope_uses_actual_peaks_and_hides_without_animation(self):
        self.meter.resize(360, 80)
        self.meter.show()
        self.app.processEvents()
        with patch("voice_changer.ui.audio_wave.monotonic", side_effect=(1.0, 1.05, 1.10)):
            self.meter.set_peak(0.01)
            self.meter.set_peak(0.1)
            self.meter._sample()
            self.meter.set_peak(1.0)
            self.meter._sample()
        self.assertEqual([peak for _, peak in self.meter.history], [0.01, 0.1, 1.0])
        image = self.meter.canvas.grab().toImage()
        blue_pixels = sum(
            1
            for y in range(image.height())
            for x in range(image.width())
            if (color := image.pixelColor(x, y)).blue() > color.red() + 30
            and color.blue() > color.green() + 30
        )
        self.assertGreater(blue_pixels, 0)
        QTest.qWait(120)
        self.assertEqual(image, self.meter.canvas.grab().toImage())
        self.assertTrue(self.meter._timer.isActive())
        self.meter.set_analysis(AnalysisReading((0.01, 0.1, 0.02), 220))
        self.meter.hide()
        self.assertFalse(self.meter._timer.isActive())
        self.assertFalse(self.meter._analysis_timer.isActive())
        self.assertFalse(self.meter._redraw_gate._timer.isActive())
        self.assertEqual(self.meter.history, ())
        self.assertEqual(self.meter.pitch_history, ())
        self.assertIsNone(self.meter.analysis_canvas.reading)
        self.meter.set_idle()
        self.meter.show()
        self.app.processEvents()
        self.assertFalse(self.meter._timer.isActive())
        self.assertEqual(self.meter.history, ())

    def test_analysis_states_values_and_accessibility_are_explicit(self):
        meter = self.meter
        self.assertIn("No capture", meter.analysis_status.text())
        meter.set_waiting()
        self.assertIn("Waiting", meter.analysis_status.text())
        meter.set_analysis_info(8000, 2)
        meter.set_analysis(AnalysisReading((0.0, 0.1, None), None, silent=True))
        meter._sample()
        self.assertIn("silence", meter.analysis_status.text())
        self.assertIn("skipped 2 blocks", meter.analysis_status.text())
        self.assertIn("4000 Hz", meter.analysis_status.text())
        description = meter.analysis_canvas.accessibleDescription()
        self.assertIn("−∞ dBFS", description)
        self.assertIn("-20.0 dBFS", description)
        self.assertIn("N/A", description)
        self.assertEqual(meter.pitch_history[-1][1], None)

        meter.set_analysis(AnalysisReading((0.01, 0.1, 1.0), None))
        self.assertIn("unvoiced", meter.analysis_status.text())
        meter.set_analysis(AnalysisReading(None, None, error="synthetic analyzer failure"))
        self.assertIn("synthetic analyzer failure", meter.analysis_status.text())
        self.assertEqual(meter.analysis_status.property("state"), "fault")
        self.assertEqual(meter.pitch_history, ())
        for reading in (
            AnalysisReading((-1.0, 0.1, 0.1), 220),
            AnalysisReading((float("nan"), 0.1, 0.1), 220),
            AnalysisReading((0.1, float("inf"), 0.1), 220),
            AnalysisReading((0.1, 0.1, 0.1), float("nan")),
            AnalysisReading((0.1, 0.1, 0.1), 801),
        ):
            with self.subTest(reading=reading):
                meter.set_analysis(reading)
                self.assertIn(ANALYSIS_INVALID, meter.analysis_status.text())
                self.assertEqual(meter.analysis_status.property("state"), "fault")
                self.assertEqual(meter.pitch_history, ())
        meter.set_idle()
        self.assertIn("No capture", meter.analysis_status.text())
        self.assertNotIn("skipped", meter.analysis_status.text())

    def test_pitch_history_is_bounded_and_unvoiced_results_leave_gaps(self):
        meter = self.meter
        with patch("voice_changer.ui.audio_wave.monotonic") as clock:
            for index in range(50):
                clock.return_value = index * 0.11
                pitch = None if index % 7 == 0 else 200.0 + index
                meter.set_analysis(AnalysisReading((0.01, 0.1, 0.01), pitch))
                meter._sample()
            history = meter.pitch_history
            self.assertLessEqual(len(history), MAX_PITCH_POINTS)
            self.assertGreaterEqual(history[0][0], history[-1][0] - WINDOW_SECONDS)
            self.assertEqual(tuple(t for t, _ in history), tuple(sorted(t for t, _ in history)))
            self.assertTrue(any(pitch is None for _, pitch in history))
            self.assertEqual(history[-2][1], 248.0)
            self.assertIsNone(history[-1][1])
            meter._sample()
            self.assertEqual(meter.pitch_history, history)
        meter.set_analysis_waiting()
        self.assertEqual(meter.pitch_history, ())
        self.assertIn("Waiting", meter.analysis_status.text())

    def test_wave_and_text_contrast_meet_the_recorded_thresholds(self):
        def luminance(color):
            channels = [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
            linear = [
                value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                for value in channels
            ]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
        for token, surface, minimum in (
            ("active", "surfaceRaised", 3.0),
            ("active", "meterTrack", 3.0),
            ("muted", "surface", 4.5),
            ("text", "surface", 4.5),
            ("danger", "dangerSurface", 4.5),
            ("active", "activeSurface", 4.5),
            ("activeInk", "active", 4.5),
        ):
            with self.subTest(token=token, surface=surface):
                foreground, background = luminance(COLORS[token]), luminance(COLORS[surface])
                ratio = (max(foreground, background) + 0.05) / (min(foreground, background) + 0.05)
                self.assertGreaterEqual(ratio, minimum)

    def test_dynamic_status_text_respects_the_state_border_and_padding(self):
        label = QLabel("synthetic capture failure")
        self.addCleanup(label.close)
        label.setStyleSheet(stylesheet())
        label.resize(300, 60)
        label.show()
        self.app.processEvents()
        for state in ("fault", "recording", "active", ""):
            with self.subTest(state=state):
                set_visual_state(label, state)
                self.app.processEvents()
                if state:
                    option = QStyleOption()
                    option.initFrom(label)
                    styled_contents = label.style().subElementRect(
                        QStyle.SubElement.SE_FrameContents, option, label,
                    )
                    self.assertEqual(label.contentsRect(), styled_contents)
                    self.assertGreater(label.contentsRect().left(), 3)
                else:
                    self.assertEqual(label.contentsRect(), label.rect())

    def test_ten_second_update_bound_for_both_lanes_and_all_graphs(self):
        second = AudioWave("处理后 / Processed")
        comparison = WaveComparison(self.meter, second)
        self.comparison = comparison
        self.addCleanup(comparison.close)
        comparison.resize(640, 440)
        comparison.show()
        self.app.processEvents()
        meters = (self.meter, second)
        ticks = [QSignalSpy(meter._timer.timeout) for meter in meters]
        requests = []
        targets = tuple(
            canvas for meter in meters for canvas in (meter.canvas, meter.analysis_canvas)
        )
        paints = {canvas: 0 for canvas in targets}

        class PaintWatcher(QObject):
            def eventFilter(self, watched, event):
                if event.type() == QEvent.Type.Paint:
                    paints[watched] += 1
                return False

        watcher = PaintWatcher()
        for canvas in targets:
            canvas.installEventFilter(watcher)

        index = 0
        injector = QTimer(comparison)
        injector.setInterval(50)

        def inject():
            nonlocal index
            for offset, meter in enumerate(meters):
                peak = 0.01 + (index % 100) * 0.001 + offset * 0.1
                meter.set_peak(peak)
                if index % 2 == 0:
                    meter.set_analysis(AnalysisReading((peak, peak, peak), 200.0 + index % 100))
            index += 1

        injector.timeout.connect(inject)
        with ExitStack() as patches:
            for widget in (comparison, *targets):
                original_update = widget.update

                def count_update(*args, original=original_update):
                    requests.append(time.monotonic())
                    original(*args)

                patches.enter_context(patch.object(widget, "update", side_effect=count_update))
            start = time.monotonic()
            injector.start()
            QTest.qWait(10000)
            injector.stop()
            bounded_requests = sum(timestamp - start <= 10 for timestamp in requests)
            sample_ticks = tuple(tick.count() for tick in ticks)
            self.assertTrue(wait_until(lambda: all(
                meter._pending_peak is None and meter._pending_analysis is None for meter in meters
            ) and not comparison._redraw_gate._timer.isActive()))
            history = tuple(meter.history for meter in meters)
            requests_without_new_data = len(requests)
            QTest.qWait(250)
            self.assertEqual(tuple(meter.history for meter in meters), history)
            self.assertEqual(len(requests), requests_without_new_data)
        self.assertGreater(index, 100)
        self.assertLessEqual(bounded_requests, 300)
        self.assertGreater(bounded_requests, 0)
        self.assertTrue(all(count > 0 for count in paints.values()), paints)
        print(
            "W04: 10 s with both lanes and peak/pitch/band updates: "
            f"{sample_ticks[0]}/{sample_ticks[1]} sample ticks, "
            f"{bounded_requests} shared repaint requests"
        )


if __name__ == "__main__":
    unittest.main()
