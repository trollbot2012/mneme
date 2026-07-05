#!/usr/bin/env sh
# Mneme installer (macOS / Linux / Git Bash). User-space only, no sudo.
#   curl -fsSL https://raw.githubusercontent.com/trollbot2012/mneme/master/install.sh | sh
set -e

REPO="${MNEME_REPO:-https://raw.githubusercontent.com/trollbot2012/mneme/master}"
DIR="${MNEME_HOME:-$HOME/.mneme}"
BIN="$HOME/.local/bin"

# locate Python 3.11+
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
      PY=$(command -v "$c"); break
    fi
  fi
done
[ -n "$PY" ] || { echo "ERROR: Python 3.11+ not found on PATH"; exit 1; }

mkdir -p "$DIR" "$BIN"
echo "downloading Mneme -> $DIR"
curl -fsSL "$REPO/mneme.py"       -o "$DIR/mneme.py"
curl -fsSL "$REPO/AGENT_SETUP.md" -o "$DIR/AGENT_SETUP.md"
curl -fsSL "$REPO/HANDOFF.md"     -o "$DIR/HANDOFF.md"
curl -fsSL "$REPO/README.md"      -o "$DIR/README.md"

# launcher
{
  echo "#!/usr/bin/env sh"
  echo "exec \"$PY\" \"$DIR/mneme.py\" \"\$@\""
} > "$BIN/mneme"
chmod +x "$BIN/mneme"

# self-test: write one note, recall it, in a throwaway dir
TMP="$DIR/.selftest.$$"
"$PY" "$DIR/mneme.py" --dir "$TMP" add --kind lesson --title "Install self-test note" --body "installer verification" >/dev/null
"$PY" "$DIR/mneme.py" --dir "$TMP" recall "install self test" | grep -q "Install self-test note" \
  || { echo "ERROR: self-test recall failed"; exit 1; }
rm -rf "$TMP"

echo ""
echo "  Mneme installed."
echo "  engine : $DIR/mneme.py   (import it, or vendor it into your agent)"
echo "  cli    : $BIN/mneme      (add PATH entry if needed: export PATH=\"\$HOME/.local/bin:\$PATH\")"
echo "  try    : mneme --dir ~/.mneme/data add --kind lesson --title \"my first note\""
echo ""
echo "  To wire it into your AI agent: give your agent the file"
echo "  $DIR/HANDOFF.md  (it contains its own instructions)"
