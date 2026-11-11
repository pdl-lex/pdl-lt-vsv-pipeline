#!/usr/bin/env bash
# Install pdl-lt-vsv-pipeline so it runs later without internet access.
#
# Run as root on the analysis machine while it still has internet:
#
#   curl -LsSf https://raw.githubusercontent.com/pdl-lex/pdl-lt-vsv-pipeline/main/scripts/setup_offline.sh \
#     | sudo bash -s -- <gruppe> [/opt]
#
# Uses the same conventions as zeitenwende-sgns/scripts/setup_offline.sh
# (same prefix, same uv-managed Python under <prefix>/uv/python), so both
# installs can live side by side. Safe to run again: it pulls and re-syncs.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/pdl-lex/pdl-lt-vsv-pipeline.git}"
NAME="pdl-lt-vsv-pipeline"

GROUP="${1:?Aufruf: setup_offline.sh <gruppe> [prefix]}"
PREFIX="${2:-/opt}"
DEST="$PREFIX/$NAME"

[[ $EUID -eq 0 ]] || { echo "Bitte als root ausführen (sudo)." >&2; exit 1; }
getent group "$GROUP" >/dev/null || { echo "Gruppe '$GROUP' existiert nicht." >&2; exit 1; }
for cmd in git curl; do
    command -v "$cmd" >/dev/null || { echo "'$cmd' fehlt." >&2; exit 1; }
done

# Python goes under the prefix (readable for all users), not into /root.
export UV_PYTHON_INSTALL_DIR="$PREFIX/uv/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy
unset UV_OFFLINE UV_NO_SYNC

echo "==> uv"
if [[ ! -x /usr/local/bin/uv ]]; then
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi
UV=/usr/local/bin/uv

echo "==> Repo nach $DEST"
mkdir -p "$PREFIX"
if [[ -d "$DEST/.git" ]]; then
    git -C "$DEST" pull --ff-only
else
    git clone "$REPO_URL" "$DEST"
fi
git config --system --get-all safe.directory 2>/dev/null | grep -qxF "$DEST" \
    || git config --system --add safe.directory "$DEST"
git -C "$DEST" config core.sharedRepository group

cd "$DEST"

echo "==> Python-Umgebung (uv sync --locked)"
"$UV" sync --locked

echo "==> Sprachmodelle nach assets/"
"$UV" run --locked python scripts/fetch_assets.py

echo "==> Offline-Test der Modelle"
UV_OFFLINE=1 UV_NO_SYNC=1 "$UV" run python scripts/fetch_assets.py --check

"$UV" cache clean

# Input data is not in the repo; create the folder so it can be dropped in.
mkdir -p stages/01_preprocessing/input

echo "==> Rechte für Gruppe '$GROUP'"
chgrp -R "$GROUP" "$DEST"
chmod -R g+rwX "$DEST"
find "$DEST" -type d -exec chmod g+s {} +
chmod -R a+rX "$PREFIX/uv"

echo "==> /etc/profile.d/pdl-lt-vsv-pipeline.sh"
cat > /etc/profile.d/pdl-lt-vsv-pipeline.sh <<EOF
# pdl-lt-vsv-pipeline: use the preinstalled environment, never go online
export UV_PYTHON_INSTALL_DIR="$PREFIX/uv/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_OFFLINE=1
export UV_NO_SYNC=1
EOF

cat <<EOF

Fertig. Nächste Schritte:
  1. Neu einloggen (oder: source /etc/profile.d/pdl-lt-vsv-pipeline.sh)
  2. Eingabedaten nach $DEST/stages/01_preprocessing/input/ kopieren
  3. cd $DEST/stages/01_preprocessing && uv run marimo edit pipeline_pre_ppmi.py
EOF
