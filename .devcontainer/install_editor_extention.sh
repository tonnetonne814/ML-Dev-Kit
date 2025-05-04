#!/usr/bin/env bash
set -euo pipefail

# === 1. List of extensions to install ========================================
extensions=(
  # --- Code‑assist / productivity -------------------------------------------
  ms-python.python                # python基本拡張機能
  ms-python.vscode-pylance        # python基本拡張機能
  ms-python.debugpy               # python基本拡張機能
  ms-python.black-formatter       # python基本拡張機能
  saoudrizwan.claude-dev          # AI コード生成
  usernamehw.errorlens            # エラー可視化
  ms-pyright.pyright              # Pythonコード向け静的型チェッカー
  charliermarsh.ruff              # コードの品質保持
  shardulm94.trailing-spaces      # 無駄なスペースの可視化
  mhutchie.git-graph              # gitコミットの可視化
  mosapride.zenkaku               # 全角スペースの可視化
  kevinrose.vsc-python-indent     # インデント自動調整
  gruntfuggly.todo-tree           # TODO管理
  aaron-bond.better-comments      # コメントの着色
  njpwerner.autodocstring         # docstring テンプレートを生成
  GitHub.copilot  # Copilot拡張
  GitHub.copilot-chat # Copilot拡張
)

# === 2. Check which CLI binaries are available ===============================
has_cursor=false; command -v cursor &>/dev/null && has_cursor=true
has_code=false;   command -v code   &>/dev/null && has_code=true

if ! $has_cursor && ! $has_code; then
  cat >&2 <<'EOF'
❌  Neither 'cursor' nor 'code' is in your PATH.
    • Cursor: Preferences → Misc → “Install ‘cursor’ command in PATH”
    • VS Code: Command Palette → “Shell Command: Install ‘code’ command in PATH”
EOF
  exit 1
fi

# === 3. Decide which CLI to use (interactive if both exist) ==================
if [ -z "${CLI:-}" ]; then          # respect pre‑set $CLI in CI environments
  if $has_cursor && $has_code; then
    echo "Both Cursor and VS Code detected. Choose which CLI to use:"
    select choice in "Cursor (cursor)" "VS Code (code)"; do
      case $REPLY in
        1) CLI=cursor; break ;;
        2) CLI=code;   break ;;
        *) echo "Invalid selection — please choose 1 or 2." ;;
      esac
    done
  elif $has_cursor; then
    CLI=cursor
  else
    CLI=code
  fi
fi
echo "▶  Target CLI: $CLI"
echo

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
