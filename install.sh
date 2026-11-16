#!/usr/bin/env bash
# Prepare a runnable environment for the voice changer demo.
#
# Creates an in-repo virtual environment that reuses the system Qt, NumPy,
# Rubber Band and PortAudio, or delegates to scripts/deploy_demo.py when a
# standalone copy is wanted. It never starts audio, never changes PipeWire,
# system volume or routing, and never installs system packages.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0
WITH_AUDIO=0
DEPLOY_TO=""

usage() {
  cat <<'USAGE'
Usage: ./install.sh [options]

Options:
  --venv DIR        virtual environment location (default: ./.venv)
  --python PATH     interpreter used to create the environment
  --with-audio      also pip install the hash-pinned audio requirements
                    (sounddevice/CFFI; needs network access)
  --deploy-to DIR   copy the demo into DIR and set it up there
                    (delegates to scripts/deploy_demo.py)
  --dry-run         print the commands, change nothing
  -h, --help        show this message

The script only prepares files and a Python environment. It does not start
audio, change system audio settings, write a desktop entry or run as root.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --venv)
      [ "$#" -ge 2 ] || { echo "--venv needs a directory" >&2; exit 2; }
      VENV_DIR="$2"; shift 2 ;;
    --python)
      [ "$#" -ge 2 ] || { echo "--python needs a path" >&2; exit 2; }
      PYTHON_BIN="$2"; shift 2 ;;
    --with-audio)
      WITH_AUDIO=1; shift ;;
    --deploy-to)
      [ "$#" -ge 2 ] || { echo "--deploy-to needs a directory" >&2; exit 2; }
      DEPLOY_TO="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    "$@"
  fi
}

step() { echo; echo "==> $*"; }

step "1/4 Checking the system"
if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script targets Debian-based systems; apt-get was not found." >&2
  exit 1
fi
if [ "$(id -u)" -eq 0 ]; then
  echo "Do not run this script as root; it only needs your user account." >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is required." >&2
  exit 1
fi
echo "Interpreter: $("$PYTHON_BIN" -V 2>&1)"
echo "Project:     $APP_DIR"

if [ -n "$DEPLOY_TO" ]; then
  step "2/4 Copying the demo to $DEPLOY_TO"
  deploy_args=(--destination "$DEPLOY_TO" --python "$PYTHON_BIN")
  if [ "$WITH_AUDIO" -eq 1 ]; then
    deploy_args+=(--install-audio)
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    deploy_args+=(--dry-run)
  fi
  run "$PYTHON_BIN" "$APP_DIR/scripts/deploy_demo.py" "${deploy_args[@]}"
  step "3/4 Skipped: the copy has its own environment"
else
  step "2/4 Creating $VENV_DIR"
  if [ -e "$VENV_DIR" ] && [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
    echo "$VENV_DIR exists and is not a virtual environment; refusing to touch it." >&2
    exit 1
  fi
  if [ -f "$VENV_DIR/pyvenv.cfg" ]; then
    echo "$VENV_DIR already exists; reusing it"
  else
    run "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
  fi

  step "3/4 Registering the source tree"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] register src/ on $VENV_DIR site-packages"
  else
    "$PYTHON_BIN" - "$APP_DIR" "$VENV_DIR" <<'PY'
import sys
from pathlib import Path

app_dir, venv_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(app_dir / "scripts"))
import deploy_demo
deploy_demo.register_source(venv_dir, app_dir)
print(f"Registered {app_dir / 'src'} in {venv_dir}")
PY
  fi
fi

if [ "$WITH_AUDIO" -eq 1 ] && [ -z "$DEPLOY_TO" ]; then
  step "Installing hash-pinned audio requirements"
  run "$VENV_DIR/bin/python" -m pip install --require-hashes \
      -r "$APP_DIR/requirements-m0-a.txt"
fi

step "4/4 Verifying the installation"
if [ -n "$DEPLOY_TO" ]; then
  check_python="$DEPLOY_TO/venv/bin/python"
else
  check_python="$VENV_DIR/bin/python"
fi
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] $check_python -B -m voice_changer --version"
  echo
  echo "Dry run finished; nothing was changed."
  exit 0
fi
if [ -x "$check_python" ]; then
  "$check_python" -B -m voice_changer --version
else
  echo "No virtual environment interpreter at $check_python" >&2
  exit 1
fi

cat <<EOF

Deployment complete

The window can be started without changing any system audio setting:
  $check_python -B -m voice_changer

Only the Python environment and project files were prepared. Audio is not
started, and no PipeWire, volume or routing setting was modified.
EOF
