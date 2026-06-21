@echo off
setlocal EnableExtensions
cd /d "%~dp0"

call :find_python
if errorlevel 1 (
    echo.
    echo [ERROR] Python 3 was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo During setup, enable "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

:menu
cls
echo ============================================================
echo                         KITSUNE
echo ============================================================
echo Python: %PYTHON_CMD%
echo Project: %CD%
echo.
echo [1] Install dependencies and cache local Qwen
echo [2] Open Streamlit
echo [3] Exit
echo.
set "CHOICE="
set /p "CHOICE=Choose an option: "

if "%CHOICE%"=="1" goto install
if "%CHOICE%"=="2" goto start
if "%CHOICE%"=="3" goto end

echo.
echo Invalid option. Choose 1, 2, or 3.
timeout /t 2 /nobreak >nul
goto menu

:install
cls
echo ============================================================
echo Installing Kitsune
echo ============================================================
echo.
echo [1/3] Updating pip...
call %PYTHON_CMD% -m pip install --upgrade pip
if errorlevel 1 goto install_failed

echo.
echo [2/3] Installing Python dependencies...
call %PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 goto install_failed

echo.
echo [3/3] Caching Qwen/Qwen2.5-0.5B-Instruct...
echo This is a one-time model download and can take several minutes.
call %PYTHON_CMD% -c "from transformers import AutoTokenizer, AutoModelForCausalLM; m='Qwen/Qwen2.5-0.5B-Instruct'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"
if errorlevel 1 (
    echo.
    echo [WARNING] Dependencies are installed, but Qwen was not cached.
    echo Kitsune will still run with its guarded local fallback.
    echo Check your internet connection and run option 1 again for local Qwen.
    echo.
    pause
    goto menu
)

echo.
echo [OK] Kitsune and local Qwen are ready.
echo.
pause
goto menu

:install_failed
echo.
echo [ERROR] Installation failed. Review the message above.
echo You can rerun option 1 after fixing the reported issue.
echo.
pause
goto menu

:start
cls
call %PYTHON_CMD% -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Streamlit is not installed.
    echo Run option 1 first.
    echo.
    pause
    goto menu
)

set "PORT="
for /f %%P in ('powershell -NoProfile -Command "$p=8501; while (Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue) { $p++ }; $p"') do set "PORT=%%P"
if not defined PORT set "PORT=8501"

set "HOST_IP="
for /f %%I in ('powershell -NoProfile -Command "$ip=(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue ^| Where-Object { $_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown' } ^| Select-Object -First 1 -ExpandProperty IPAddress); if($ip){$ip}else{'127.0.0.1'}"') do set "HOST_IP=%%I"
if not defined HOST_IP set "HOST_IP=127.0.0.1"

echo ============================================================
echo Starting Kitsune
echo ============================================================
echo.
echo Local link:   http://localhost:%PORT%
echo Network link: http://%HOST_IP%:%PORT%
echo.
echo Keep this window open while using the app.
echo Press Ctrl+C here to stop the server.
echo ============================================================
echo.

start "" "http://localhost:%PORT%"
call %PYTHON_CMD% -m streamlit run app.py --server.address=0.0.0.0 --server.port=%PORT% --server.headless=true --browser.gatherUsageStats=false

echo.
echo Streamlit stopped.
pause
goto menu

:find_python
where py >nul 2>nul
if not errorlevel 1 (
    py -3 --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
        exit /b 0
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    python --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        exit /b 0
    )
)

exit /b 1

:end
endlocal
exit /b 0
