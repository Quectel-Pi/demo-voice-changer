"""PipeWire virtual microphone support; no audio I/O and no daemon changes.

The processed output is played into a private PipeWire sink. A ``pw-loopback``
process forwards that sink into a source node, which other applications can
select as a microphone. ALSA reaches the private sink through a generated
``type pipewire`` PCM that names the node explicitly.

The ALSA ``pipewire`` plugin opens successfully even when the target node does
not exist, and then discards every sample. Node presence is therefore checked
explicitly instead of trusting a successful stream open.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from time import monotonic, sleep

PCM_NAME = "vcmic"
SINK_NAME = "voice-changer-sink"
SOURCE_NAME = "voice-changer-mic"
DESCRIPTION = "Voice Changer Virtual Mic"

# The PCM id is also the PortAudio device name, so it must stay recognizable.
_PCM_PATTERN = re.compile(r"\A[A-Za-z][A-Za-z0-9_]*\Z")
_NODE_PATTERN = re.compile(r"\A[A-Za-z][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class Support:
    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise ValueError("available must be a boolean")
        if self.available and self.reason is not None:
            raise ValueError("Available support must not carry a failure reason")
        if not self.available and not self.reason:
            raise ValueError("Unavailable support must explain why")


def alsa_configuration(
    *, pcm: str = PCM_NAME, sink: str = SINK_NAME, source: str = SOURCE_NAME,
    description: str = DESCRIPTION,
) -> str:
    """Return an ALSA fragment that routes one PCM to explicit PipeWire nodes."""
    for value, field, pattern in (
        (pcm, "PCM", _PCM_PATTERN), (sink, "Sink node", _NODE_PATTERN),
        (source, "Source node", _NODE_PATTERN),
    ):
        if not isinstance(value, str) or not pattern.match(value):
            raise ValueError(f"{field} must be a simple ALSA/PipeWire identifier")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Description must be explicit")
    escaped = description.replace("\\", "\\\\").replace('"', '\\"')
    if "\n" in escaped or "\r" in escaped:
        raise ValueError("Description cannot contain line breaks")
    return (
        "</usr/share/alsa/alsa.conf>\n"
        f"pcm.{pcm} {{\n"
        "    type pipewire\n"
        f'    playback_node "{sink}"\n'
        f'    capture_node "{source}"\n'
        "    rate 48000\n"
        "    channels 1\n"
        "    hint {\n"
        "        show on\n"
        f'        description "{escaped}"\n'
        "    }\n"
        "}\n"
    )


def probe_support(*, environ=None, which=None) -> Support:
    """Report whether this session can host a PipeWire virtual microphone."""
    environ = os.environ if environ is None else environ
    which = shutil.which if which is None else which
    if not environ.get("XDG_RUNTIME_DIR"):
        return Support(False, "XDG_RUNTIME_DIR is unset; no user PipeWire session")
    missing = [tool for tool in ("pw-loopback", "pw-dump") if which(tool) is None]
    if missing:
        return Support(False, f"missing PipeWire tools: {', '.join(missing)}")
    socket = Path(environ["XDG_RUNTIME_DIR"]) / "pipewire-0"
    if not socket.exists():
        return Support(False, f"PipeWire socket not found: {socket}")
    return Support(True)


@dataclass(frozen=True)
class Availability:
    """Outcome of preparing the virtual PCM for this process."""

    support: Support
    config_path: Path | None = None

    @property
    def available(self) -> bool:
        return self.support.available and self.config_path is not None

    @property
    def reason(self) -> str | None:
        if self.support.reason is not None:
            return self.support.reason
        if self.config_path is None:
            return "could not write the ALSA virtual PCM configuration"
        return None


def capabilities_or_none() -> Availability | None:
    """Read the startup result without probing; ``None`` means startup did not run."""
    return _prepared


_prepared: Availability | None = None


def prepare(*, environ=None) -> Availability:
    """Install the ALSA fragment once per process, before sounddevice is imported.

    PortAudio caches the ALSA device list when it first initializes, so the
    fragment is written during startup rather than on first device use.
    """
    global _prepared
    if _prepared is None:
        environ = os.environ if environ is None else environ
        path = install_alsa_config(environ=environ)
        _prepared = Availability(probe_support(environ=environ), path)
    return _prepared


def reset_prepared() -> None:
    """Forget the startup result; tests only, never used by the application."""
    global _prepared
    _prepared = None


def capabilities() -> Availability:
    """Return the startup result; never probes or writes after startup."""
    if _prepared is None:
        raise RuntimeError("Virtual microphone support was not prepared during startup")
    return _prepared


def install_alsa_config(*, target: Path | None = None, environ=None) -> Path | None:
    """Write the fragment and point ALSA at it; must run before sounddevice loads.

    Returns the path on success and ``None`` when the virtual PCM cannot be
    offered in this session. Failure is never silent: the caller reports it.
    """
    environ = os.environ if environ is None else environ
    support = probe_support(environ=environ)
    if not support.available:
        return None
    runtime = Path(environ["XDG_RUNTIME_DIR"])
    directory = runtime / "voice-changer"
    path = Path(target) if target is not None else directory / "asound.conf"
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        text = alsa_configuration()
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
    except OSError:
        return None
    environ["ALSA_CONFIG_PATH"] = str(path)
    return path


def _node_names(document: object) -> set[str]:
    if not isinstance(document, list):
        raise ValueError("PipeWire dump must be a list")
    names: set[str] = set()
    for entry in document:
        if not isinstance(entry, dict):
            continue
        info = entry.get("info")
        if not isinstance(info, dict):
            continue
        props = info.get("props")
        if not isinstance(props, dict):
            continue
        name = props.get("node.name")
        if isinstance(name, str):
            names.add(name)
    return names


def present_nodes(*, timeout: float = 5.0, runner=subprocess.run) -> set[str]:
    """Read current PipeWire node names with a bounded subprocess call."""
    result = runner(
        ["pw-dump"], capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pw-dump failed with status {result.returncode}: {result.stderr.strip()[:200]}"
        )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("pw-dump did not return JSON") from error
    return _node_names(document)


def virtual_nodes_ready(*, timeout: float = 5.0, poll: float = 0.1, reader=present_nodes) -> bool:
    """Wait until both loopback nodes are visible or report that they never were."""
    deadline = monotonic() + timeout
    while True:
        names = reader()
        if SINK_NAME in names and SOURCE_NAME in names:
            return True
        if monotonic() >= deadline:
            return False
        sleep(poll)


class VirtualMicSession:
    """Own one ``pw-loopback`` child; safe to stop more than once."""

    def __init__(
        self, *, popen=subprocess.Popen, ready=virtual_nodes_ready, environ=None,
    ) -> None:
        self._popen = popen
        self._ready = ready
        self._environ = environ
        self._process = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Virtual microphone session already started")
        if not probe_support(environ=self._environ).available:
            raise RuntimeError(
                "Virtual microphone needs a running PipeWire session with pw-loopback and pw-dump"
            )
        process = self._popen(
            [
                "pw-loopback", "-n", "voice-changer-virtual-mic",
                f"--capture-props=media.class=Audio/Sink node.name={SINK_NAME} "
                f"node.description=Voice-changer-output audio.position=[MONO] stream.dont-remix=true",
                f"--playback-props=media.class=Audio/Source node.name={SOURCE_NAME} "
                f"node.description=Voice-Changer-Virtual-Mic audio.position=[MONO] "
                "stream.dont-remix=true",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        self._process = process
        try:
            if not self._ready():
                detail = ""
                if process.poll() is not None:
                    detail = f"; pw-loopback exited with {process.returncode}"
                    if process.stderr is not None:
                        detail += f": {process.stderr.read()[:300].strip()}"
                raise RuntimeError(
                    f"PipeWire did not publish the virtual microphone nodes{detail}"
                )
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()
        self._process = None
