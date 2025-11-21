@echo off
rem =============================================================
rem  VS Code / Cursor extension installer  —  run from cmd.exe
rem =============================================================
setlocal EnableExtensions EnableDelayedExpansion

rem -- Detect CLI -------------------------------------------------
set "CLI="
set "HAS_CURSOR=0"
set "HAS_CODE=0"
set "HAS_ANTIGRAVITY=0"

where cursor >nul 2>&1 && set "HAS_CURSOR=1"
where code   >nul 2>&1 && set "HAS_CODE=1"
where antigravity >nul 2>&1 && set "HAS_ANTIGRAVITY=1"

rem -- Interactive Selection ---------------------------------------
set "COUNT=0"
if "%HAS_CURSOR%"=="1" set /a COUNT+=1
if "%HAS_CODE%"=="1" set /a COUNT+=1
if "%HAS_ANTIGRAVITY%"=="1" set /a COUNT+=1

if "%COUNT%"=="0" (
    echo Neither cursor, code, nor antigravity command found in PATH.
    goto :eof
)

if "%COUNT%"=="1" (
    if "%HAS_CURSOR%"=="1" set "CLI=cursor"
    if "%HAS_CODE%"=="1" set "CLI=code"
    if "%HAS_ANTIGRAVITY%"=="1" set "CLI=antigravity"
) else (
    echo Multiple editors detected. Choose which CLI to use:
    if "%HAS_CURSOR%"=="1" echo [1] Cursor ^(cursor^)
    if "%HAS_CODE%"=="1" echo [2] VS Code ^(code^)
    if "%HAS_ANTIGRAVITY%"=="1" echo [3] Antigravity ^(antigravity^)
    
    set /p "CHOICE=Enter number: "
    
    if "!CHOICE!"=="1" if "%HAS_CURSOR%"=="1" set "CLI=cursor"
    if "!CHOICE!"=="2" if "%HAS_CODE%"=="1" set "CLI=code"
    if "!CHOICE!"=="3" if "%HAS_ANTIGRAVITY%"=="1" set "CLI=antigravity"
    
    if not defined CLI (
        echo Invalid selection. Exiting.
        goto :eof
    )
)

echo Target CLI: !CLI!

rem -- List of extensions ----------------------------------------
for %%E in (
    ms-python.python
    ms-python.vscode-pylance
    ms-python.debugpy
    ms-python.black-formatter
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
