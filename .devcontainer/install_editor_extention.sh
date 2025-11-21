#!/usr/bin/env bash
set -euo pipefail

# === 1. List of extensions to install ========================================
extensions=(
  # --- Code‑assist / productivity -------------------------------------------
  ms-python.python                    # Python基本拡張機能
  ms-python.vscode-pylance            # Python基本拡張機能
  ms-python.debugpy                   # Python基本拡張機能
  ms-python.black-formatter           # Python基本拡張機能
  ms-azuretools.vscode-docker         # Docker基本拡張機能
  ms-vscode-remote.remote-containers  # Docker基本拡張機能
  ms-vscode-remote.remote-ssh         # remote基本拡張機能
  ms-vscode.remote-server             # remote基本拡張機能
  usernamehw.errorlens                # エラー可視化
  ms-pyright.pyright                  # Pythonコード向け静的型チェッカー
  charliermarsh.ruff                  # コードの品質保持
  shardulm94.trailing-spaces          # 無駄なスペースの可視化
  mhutchie.git-graph                  # gitコミットの可視化
  mosapride.zenkaku                   # 全角スペースの可視化
  kevinrose.vsc-python-indent         # インデント自動調整
  gruntfuggly.todo-tree               # TODO管理
  aaron-bond.better-comments          # コメントの着色
  njpwerner.autodocstring             # docstring テンプレートを生成
)

# === 2. Check which CLI binaries are available ===============================
has_cursor=false; command -v cursor &>/dev/null && has_cursor=true
has_code=false;   command -v code   &>/dev/null && has_code=true
has_antigravity=false; command -v antigravity &>/dev/null && has_antigravity=true

if ! $has_cursor && ! $has_code && ! $has_antigravity; then
  cat >&2 <<'EOF'
❌  Neither 'cursor', 'code', nor 'antigravity' is in your PATH.
    • Cursor: Preferences → Misc → “Install ‘cursor’ command in PATH”
    • VS Code: Command Palette → “Shell Command: Install ‘code’ command in PATH”
    • Antigravity: Ensure the 'antigravity' command is available in your PATH.
EOF
  exit 1
fi

# === 3. Decide which CLI to use (interactive if both exist) ==================
if [ -z "${CLI:-}" ]; then          # respect pre‑set $CLI in CI environments
  # Count available editors
  count=0
  $has_cursor && ((count++))
  $has_code && ((count++))
  $has_antigravity && ((count++))

  if [ "$count" -gt 1 ]; then
    echo "Multiple editors detected. Choose which CLI to use:"
    options=()
    $has_cursor && options+=("Cursor (cursor)")
    $has_code && options+=("VS Code (code)")
    $has_antigravity && options+=("Antigravity (antigravity)")
    
    select choice in "${options[@]}"; do
      case $choice in
        "Cursor (cursor)") CLI=cursor; break ;;
        "VS Code (code)") CLI=code; break ;;
        "Antigravity (antigravity)") CLI=antigravity; break ;;
        *) echo "Invalid selection." ;;
      esac
    done
  elif $has_cursor; then
    CLI=cursor
  elif $has_code; then
    CLI=code
  elif $has_antigravity; then
    CLI=antigravity
  fi
fi
echo "▶  Target CLI: $CLI"
echo

if [ "$CLI" = "antigravity" ]; then
  echo "⚙️  Configuring Antigravity Marketplace..."
  SETTINGS_DIR="$HOME/.config/antigravity/User"
  mkdir -p "$SETTINGS_DIR"
  SETTINGS_FILE="$SETTINGS_DIR/settings.json"

  python3 -c "
import json
import os

settings_file = '$SETTINGS_FILE'
data = {}

if os.path.exists(settings_file):
    try:
        with open(settings_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        pass

if 'extensionsGallery' not in data:
    data['extensionsGallery'] = {}

data['extensionsGallery']['serviceUrl'] = 'https://marketplace.visualstudio.com/_apis/public/gallery'
data['extensionsGallery']['itemUrl'] = 'https://marketplace.visualstudio.com/items'

with open(settings_file, 'w') as f:
    json.dump(data, f, indent=4)
"
fi

# === 4. Install extensions (skip those already present) ======================
for ext in "${extensions[@]}"; do
  if $CLI --list-extensions | grep -qx "$ext"; then
    echo "✔  $ext (already installed)"
  else
    echo "➕  Installing $ext …"
    $CLI --install-extension "$ext"
  fi
done

echo
echo "🎉  All done!"
