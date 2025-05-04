@echo off
rem =============================================================
rem  VS Code / Cursor extension installer  —  run from cmd.exe
rem =============================================================
setlocal EnableExtensions EnableDelayedExpansion

rem -- Detect CLI -------------------------------------------------
where cursor >nul 2>&1 && set "CLI=cursor"
where  code   >nul 2>&1 && if not defined CLI set "CLI=code"

if not defined CLI (
    echo Neither cursor nor code command found in PATH.
    goto :eof
)

rem -- List of extensions ----------------------------------------
for %%E in (
    ms-python.python
    ms-python.vscode-pylance
    ms-python.debugpy
    ms-python.black-formatter
    saoudrizwan.claude-dev
    usernamehw.errorlens
    ms-pyright.pyright
    charliermarsh.ruff
    shardulm94.trailing-spaces
    mhutchie.git-graph
    mosapride.zenkaku
    kevinrose.vsc-python-indent
    gruntfuggly.todo-tree
    aaron-bond.better-comments
    njpwerner.autodocstring
    GitHub.copilot
    GitHub.copilot-chat
) do (
    %CLI% --list-extensions | findstr /I /X "%%E" >nul
    if errorlevel 1 (
        echo Installing %%E...
        %CLI% --install-extension %%E
    ) else (
        echo %%E already installed.
    )
)

echo.
echo Done!
endlocal
