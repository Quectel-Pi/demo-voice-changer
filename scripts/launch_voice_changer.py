#!/usr/bin/env python3
"""Desktop entry point independent of the caller's working directory."""

from pathlib import Path
import os
import subprocess
import sys


if __name__ == "__main__":
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = source + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-m", "voice_changer", *sys.argv[1:]], env=environment,
        ).returncode
    except OSError as error:
        print(f"变声器启动失败 / Voice changer startup failed: {error}", file=sys.stderr)
        result = 1
    if result:
        print(f"变声器退出错误 / Voice changer failed, exit status: {result}", file=sys.stderr)
        if sys.stdin.isatty():
            try:
                input("启动失败，请检查上方错误及README；按回车关闭。 / Press Enter to close.")
            except EOFError:
                print("启动终端输入已关闭 / Terminal input closed", file=sys.stderr)
    raise SystemExit(1 if result < 0 else result)
