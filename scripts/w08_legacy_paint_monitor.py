#!/usr/bin/env python3
"""Opt-in legacy paint observer; the selected app must expose a window-ready hook."""

import argparse
import ast
from collections.abc import Callable, Sequence
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from w08_paint_monitor import PaintSession


class LaunchError(Exception):
    """The selected legacy application cannot be observed safely."""


def _source_paths(value: str) -> tuple[Path, Path, Path, Path]:
    try:
        root = Path(value).expanduser().resolve(strict=True)
        package = (root / "voice_changer").resolve(strict=True)
        entry = (package / "app" / "main.py").resolve(strict=True)
        window = (package / "ui" / "main_window.py").resolve(strict=True)
        package_init = (package / "__init__.py").resolve(strict=True)
    except OSError as error:
        raise LaunchError(f"Cannot resolve the selected application source: {error}") from error
    if (
        not root.is_dir() or not package.is_dir() or not package.is_relative_to(root)
        or any(not path.is_file() or not path.is_relative_to(package)
               for path in (entry, window, package_init))
    ):
        raise LaunchError("The selected source must contain one complete voice_changer package")
    return root, package, entry, window


def _check_package_origins(package: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "voice_changer" and not name.startswith("voice_changer."):
            continue
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            raise LaunchError(f"Cannot verify the origin of module {name}")
        try:
            path = Path(origin).resolve(strict=True)
        except OSError as error:
            raise LaunchError(f"Cannot resolve the origin of module {name}: {error}") from error
        if not path.is_relative_to(package):
            raise LaunchError(f"Module {name} is outside the selected application source")
        for location in getattr(module, "__path__", ()):
            try:
                package_path = Path(location).resolve(strict=True)
            except OSError as error:
                raise LaunchError(f"Cannot resolve the package path for {name}: {error}") from error
            if not package_path.is_relative_to(package):
                raise LaunchError(f"Package {name} searches outside the selected application source")


def _check_hook_source(entry: Path) -> None:
    try:
        tree = ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise LaunchError(f"Cannot inspect the selected application entry: {error}") from error
    declarations = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    ]
    if len(declarations) != 1 or not isinstance(declarations[0], ast.FunctionDef):
        raise LaunchError("The selected entry must have one explicit synchronous main function")
    arguments = declarations[0].args
    positional = arguments.posonlyargs + arguments.args
    defaults = {
        argument.arg: default
        for argument, default in zip(
            positional[len(positional) - len(arguments.defaults):],
            arguments.defaults,
        )
    }
    defaults.update({
        argument.arg: default
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
    })
    accepted = {argument.arg for argument in arguments.args + arguments.kwonlyargs}
    if "window_ready" not in accepted:
        raise LaunchError("The selected application entry has no explicit window_ready hook")
    default = defaults.get("window_ready")
    if not isinstance(default, ast.Constant) or default.value is not None:
        raise LaunchError("The window_ready hook must be disabled by default")


def _load_entry(root: Path, package: Path, entry: Path) -> Callable[..., int]:
    if any(name == "voice_changer" or name.startswith("voice_changer.") for name in sys.modules):
        raise LaunchError("voice_changer was already imported; start the observer in a fresh process")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("voice_changer.app.main")
    except ImportError as error:
        raise LaunchError(f"Cannot import the selected application entry: {error}") from error
    _check_package_origins(package)
    if Path(module.__file__).resolve() != entry:
        raise LaunchError("The imported application entry is not the selected file")
    app_main = getattr(module, "main", None)
    if not callable(app_main):
        raise LaunchError("The selected application entry has no callable main function")
    try:
        signature = inspect.signature(app_main)
        hook = signature.parameters.get("window_ready")
        if hook is None or hook.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
        ):
            raise LaunchError("The selected application entry has no explicit window_ready hook")
        if hook.default is not None:
            raise LaunchError("The window_ready hook must be disabled by default")
        signature.bind([], window_ready=None)
    except (TypeError, ValueError) as error:
        raise LaunchError(f"Cannot use the selected window_ready hook: {error}") from error
    return app_main


def _load_observer() -> ModuleType:
    try:
        path = Path(__file__).with_name("w08_paint_monitor.py").resolve(strict=True)
    except OSError as error:
        raise LaunchError(f"Cannot resolve the paint observer: {error}") from error
    name = "_w08_legacy_paint_observer"
    if name in sys.modules:
        raise LaunchError("The legacy paint observer was already loaded")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LaunchError("Cannot load the adjacent paint observer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loaded = False
    try:
        try:
            spec.loader.exec_module(module)
        except (ImportError, OSError) as error:
            raise LaunchError(f"Cannot import the paint observer: {error}") from error
        loaded = True
    finally:
        if not loaded:
            sys.modules.pop(name, None)
    return module


def _failure(reason: str, *, root: Path | None = None) -> int:
    print(f"Legacy paint observer setup failed: {reason}", file=sys.stderr, flush=True)
    report = {
        "kind": "qt_paint_events", "status": "incomplete", "pid": os.getpid(),
        "issues": [reason], "application_source": str(root) if root is not None else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1


def _request_close_after_failure(window: object) -> str | None:
    from PySide6.QtCore import QMetaObject, Qt
    from PySide6.QtWidgets import QApplication, QWidget
    app = QApplication.instance()
    is_window = isinstance(window, QWidget)
    target = window if is_window else app
    if target is None:
        return "Cannot close after observer failure: no Qt application is available"
    try:
        queued = QMetaObject.invokeMethod(
            target, "close" if is_window else "quit", Qt.ConnectionType.QueuedConnection,
        )
    except (RuntimeError, TypeError) as error:
        return f"Cannot queue close after observer failure: {error}"
    return None if queued else "Cannot queue close after observer failure"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-real-app", action="store_true")
    parser.add_argument(
        "--app-src",
        help="Explicit source root containing the selected voice_changer package.",
    )
    args = parser.parse_args(argv)
    if not args.launch_real_app:
        parser.error("--launch-real-app is required to launch the desktop app")
    if not args.app_src:
        parser.error("--app-src is required to identify the selected application")

    print(
        "将启动指定窗口，可能发现设备并写共享 ALSA 配置；须事先获得明确授权。 / "
        "Launching the selected window may discover devices and write shared ALSA "
        "configuration; separate authorization is required.",
        file=sys.stderr, flush=True,
    )
    root: Path | None = None
    try:
        root, package, entry, window_file = _source_paths(args.app_src)
        _check_hook_source(entry)
        app_main = _load_entry(root, package, entry)
    except LaunchError as error:
        return _failure(str(error), root=root)

    sessions: list["PaintSession"] = []
    issues: list[str] = []

    def attach(window: object) -> None:
        if issues:
            return
        try:
            if sessions:
                raise LaunchError("The window-ready callback was invoked more than once")
            _check_package_origins(package)
            from PySide6.QtWidgets import QWidget
            if not isinstance(window, QWidget):
                raise LaunchError("The selected window is not a QWidget")
            window_module = sys.modules.get("voice_changer.ui.main_window")
            if (
                window.__class__.__module__ != "voice_changer.ui.main_window"
                or window_module is None
                or Path(window_module.__file__).resolve() != window_file
            ):
                raise LaunchError("The legacy window does not come from the selected source")
            observer = _load_observer()
            session = observer.PaintSession(legacy=True)
            sessions.append(session)
            session.attach(window)
        except LaunchError as error:
            issues.append(str(error))
            print(f"Legacy paint observer setup failed: {error}", file=sys.stderr, flush=True)
            close_error = _request_close_after_failure(window)
            if close_error:
                issues.append(close_error)
                print(close_error, file=sys.stderr, flush=True)

    result = app_main([], window_ready=attach)
    if not sessions and not issues:
        issues.append("Application exited before invoking the window-ready callback")
        print(f"Legacy paint observer setup failed: {issues[-1]}", file=sys.stderr, flush=True)
    report = (
        {"kind": "qt_paint_events", "status": "incomplete", "pid": os.getpid(), "issues": issues}
        if issues else sessions[0].report()
    )
    report["application_source"] = str(root)
    report["application_entry"] = str(entry)
    report["window_module"] = str(window_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return result if result else (0 if report["status"] == "complete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
