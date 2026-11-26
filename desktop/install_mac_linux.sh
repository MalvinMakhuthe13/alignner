#!/usr/bin/env bash
# Alignner as an app on macOS / Linux: makes .venv, installs, adds a launcher.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="$(command -v python3 || command -v python)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -q --upgrade pip
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
if [ "$(uname)" = "Darwin" ]; then
  APP="$HOME/Applications/Alignner.command"; mkdir -p "$HOME/Applications"
  printf '#!/bin/bash\ncd "%s" && exec .venv/bin/python desktop/alignner_desktop.py\n' "$ROOT" > "$APP"; chmod +x "$APP"
  echo "Done: open ~/Applications/Alignner.command (drag it to the Dock)."
else
  DESK="$HOME/.local/share/applications/alignner.desktop"; mkdir -p "$(dirname "$DESK")"
  cat > "$DESK" <<EOT
[Desktop Entry]
Name=Alignner
Comment=Photo alignment and point clouds
Exec=$ROOT/.venv/bin/python $ROOT/desktop/alignner_desktop.py
Icon=$ROOT/desktop/alignner.png
Terminal=false
Type=Application
Categories=Graphics;
EOT
  echo "Done: Alignner is in your applications menu."
fi
