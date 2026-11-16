"""Virtual microphone configuration and process lifetime; no device or daemon use."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from voice_changer.audio import virtual_mic


class AlsaConfigurationTests(unittest.TestCase):
    def test_fragment_names_the_pcm_and_both_explicit_nodes(self):
        text = virtual_mic.alsa_configuration()
        self.assertIn("pcm.vcmic {", text)
        self.assertIn('playback_node "voice-changer-sink"', text)
        self.assertIn('capture_node "voice-changer-mic"', text)
        self.assertIn("type pipewire", text)
        self.assertIn("show on", text)
        self.assertTrue(text.startswith("</usr/share/alsa/alsa.conf>"))
        self.assertIn("Voice Changer Virtual Mic", text)

    def test_rejects_identifiers_that_could_escape_the_fragment(self):
        for bad in ("", "1pcm", "pcm\n", "a b", 'a"', "a{", "µ"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    virtual_mic.alsa_configuration(pcm=bad)
        with self.assertRaises(ValueError):
            virtual_mic.alsa_configuration(sink="bad node")
        with self.assertRaises(ValueError):
            virtual_mic.alsa_configuration(description="line\nbreak")
        with self.assertRaises(ValueError):
            virtual_mic.alsa_configuration(description="   ")

    def test_description_quotes_are_escaped_not_dropped(self):
        text = virtual_mic.alsa_configuration(description='say "hi"')
        self.assertIn('description "say \\"hi\\""', text)


class SupportTests(unittest.TestCase):
    def test_reports_missing_runtime_directory(self):
        support = virtual_mic.probe_support(environ={}, which=lambda _tool: "/usr/bin/x")
        self.assertFalse(support.available)
        self.assertIn("XDG_RUNTIME_DIR", support.reason)

    def test_reports_missing_tools_by_name(self):
        runtime = "/run/user/1000"
        support = virtual_mic.probe_support(
            environ={"XDG_RUNTIME_DIR": runtime}, which=lambda tool: None,
        )
        self.assertFalse(support.available)
        self.assertIn("pw-loopback", support.reason)
        self.assertIn("pw-dump", support.reason)

    def test_reports_absent_socket_rather_than_claiming_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            support = virtual_mic.probe_support(
                environ={"XDG_RUNTIME_DIR": temporary}, which=lambda _tool: "/usr/bin/x",
            )
            self.assertFalse(support.available)
            self.assertIn("pipewire-0", support.reason)

    def test_reports_available_when_socket_and_tools_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "pipewire-0").touch()
            support = virtual_mic.probe_support(
                environ={"XDG_RUNTIME_DIR": temporary}, which=lambda _tool: "/usr/bin/x",
            )
            self.assertTrue(support.available)
            self.assertIsNone(support.reason)

    def test_unavailable_support_requires_a_reason(self):
        with self.assertRaises(ValueError):
            virtual_mic.Support(False)
        with self.assertRaises(ValueError):
            virtual_mic.Support(True, "why")


class AlsaConfigInstallTests(unittest.TestCase):
    def test_install_points_alsa_at_a_private_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            environ = {"XDG_RUNTIME_DIR": temporary}
            (Path(temporary) / "pipewire-0").touch()
            path = virtual_mic.install_alsa_config(
                environ=environ, target=Path(temporary) / "conf" / "asound.conf",
            )
            self.assertIsNotNone(path)
            self.assertEqual(environ["ALSA_CONFIG_PATH"], str(path))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertIn("pcm.vcmic", path.read_text(encoding="utf-8"))

    def test_install_declines_without_pipewire_and_leaves_environment_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            environ = {"XDG_RUNTIME_DIR": temporary}
            path = virtual_mic.install_alsa_config(
                environ=environ, target=Path(temporary) / "asound.conf",
            )
            self.assertIsNone(path)
            self.assertNotIn("ALSA_CONFIG_PATH", environ)
            self.assertFalse((Path(temporary) / "asound.conf").exists())


class NodeParsingTests(unittest.TestCase):
    def test_reads_node_names_from_a_dump(self):
        document = [
            {"id": 1, "info": {"props": {"node.name": "a"}}},
            {"id": 2, "info": {"props": {}}},
            {"id": 3},
            {"id": 4, "info": {"props": {"node.name": "b"}}},
        ]
        self.assertEqual(virtual_mic._node_names(document), {"a", "b"})

    def test_rejects_non_list_dump(self):
        with self.assertRaises(ValueError):
            virtual_mic._node_names({"info": {}})

    def test_present_nodes_reports_a_failed_command(self):
        def runner(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 1, "", "boom")

        with self.assertRaises(RuntimeError):
            virtual_mic.present_nodes(runner=runner)

    def test_present_nodes_rejects_invalid_json(self):
        def runner(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, "not json", "")

        with self.assertRaises(RuntimeError):
            virtual_mic.present_nodes(runner=runner)

    def test_wait_returns_false_instead_of_hanging(self):
        calls = []

        def reader():
            calls.append(1)
            return set()

        started = __import__("time").monotonic()
        self.assertFalse(virtual_mic.virtual_nodes_ready(timeout=0.05, poll=0.01, reader=reader))
        self.assertLess(__import__("time").monotonic() - started, 2.0)
        self.assertGreater(len(calls), 1)

    def test_wait_succeeds_once_both_nodes_appear(self):
        seen = []

        def reader():
            seen.append(1)
            return {"voice-changer-sink", "voice-changer-mic"} if len(seen) > 2 else set()

        self.assertTrue(virtual_mic.virtual_nodes_ready(timeout=2, poll=0.01, reader=reader))


class _FakeProcess:
    def __init__(self, *, exit_code=None, stderr="") -> None:
        self.returncode = exit_code
        self.stderr = None if stderr is None else _FakeStream(stderr)
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class _FakeStream:
    def __init__(self, text: str) -> None:
        self._text = text
        self.closed = False

    def read(self, *_args):
        return self._text

    def close(self):
        self.closed = True


class SessionTests(unittest.TestCase):
    def _session(self, process, ready, environ):
        captured = {}

        def popen(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return process

        session = virtual_mic.VirtualMicSession(
            popen=popen, ready=ready, environ=environ,
        )
        return session, captured

    def _environ(self, temporary):
        (Path(temporary) / "pipewire-0").touch()
        return {"XDG_RUNTIME_DIR": temporary}

    def test_start_passes_explicit_node_identity_and_hides_stdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = _FakeProcess()
            session, captured = self._session(process, lambda: True, self._environ(temporary))
            session.start()
            self.assertTrue(session.running)
            self.assertEqual(captured["args"][0], "pw-loopback")
            self.assertEqual(captured["kwargs"]["stdout"], subprocess.DEVNULL)
            joined = " ".join(captured["args"])
            self.assertIn("node.name=voice-changer-sink", joined)
            self.assertIn("node.name=voice-changer-mic", joined)
            self.assertIn("media.class=Audio/Sink", joined)
            self.assertIn("media.class=Audio/Source", joined)

    def test_start_fails_loudly_when_nodes_never_appear_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = _FakeProcess(exit_code=1, stderr="no session")
            session, _ = self._session(process, lambda: False, self._environ(temporary))
            with self.assertRaises(RuntimeError) as raised:
                session.start()
            self.assertIn("virtual microphone", str(raised.exception).lower())
            self.assertIn("no session", str(raised.exception))
            self.assertFalse(session.running)

    def test_start_refuses_without_a_pipewire_socket(self):
        with tempfile.TemporaryDirectory() as temporary:
            session, captured = self._session(_FakeProcess(), lambda: True, {"XDG_RUNTIME_DIR": temporary})
            with self.assertRaises(RuntimeError):
                session.start()
            self.assertNotIn("args", captured)

    def test_stop_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            process = _FakeProcess()
            session, _ = self._session(process, lambda: True, self._environ(temporary))
            session.start()
            session.stop()
            session.stop()
            self.assertTrue(process.terminated)
            self.assertFalse(session.running)

    def test_start_twice_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            session, _ = self._session(_FakeProcess(), lambda: True, self._environ(temporary))
            session.start()
            try:
                with self.assertRaises(RuntimeError):
                    session.start()
            finally:
                session.stop()


if __name__ == "__main__":
    unittest.main()
