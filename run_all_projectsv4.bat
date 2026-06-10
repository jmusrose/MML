@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "PYTHON_EXE=E:\anaconda3\envs\pytorch2.5\python.exe"
set "DRY_RUN=0"
set "CONTINUE_ON_ERROR=0"

:parse_args
if "%~1"=="" goto after_parse
if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto parse_args
)
if /I "%~1"=="-dry-run" (
    set "DRY_RUN=1"
    shift
    goto parse_args
)
if /I "%~1"=="--continue-on-error" (
    set "CONTINUE_ON_ERROR=1"
    shift
    goto parse_args
)
if /I "%~1"=="-continue-on-error" (
    set "CONTINUE_ON_ERROR=1"
    shift
    goto parse_args
)
if /I "%~1"=="--python" (
    set "PYTHON_EXE=%~2"
    shift
    shift
    goto parse_args
)
echo Unknown argument: %~1
echo Usage: run_all_projectsv4.bat [--dry-run] [--continue-on-error] [--python path\to\python.exe]
exit /b 2

:after_parse
if not exist "%PYTHON_EXE%" (
    echo Python executable not found: %PYTHON_EXE%
    echo Falling back to python on PATH.
    set "PYTHON_EXE=python"
)

call :run_step "RGB_v4 NYU" "%ROOT%RGB_v4" "DML_nyu.py"
if errorlevel 1 exit /b %errorlevel%

call :run_step "RGB_v4 SUN" "%ROOT%RGB_v4" "DML_sun.py"
if errorlevel 1 exit /b %errorlevel%

call :run_step "MVSA_v4" "%ROOT%MVSA_v4" "DML_MVSA.py"
if errorlevel 1 exit /b %errorlevel%

call :run_step "Food_v4" "%ROOT%Food_v4" "DML_Food.py"
if errorlevel 1 exit /b %errorlevel%

call :run_step "CREMAD_v4" "%ROOT%CREMAD_v4" "DML_cremad.py"
if errorlevel 1 exit /b %errorlevel%

echo.
echo All requested v4 information bottleneck and conformal project runs finished.
exit /b 0

:run_step
set "STEP_NAME=%~1"
set "STEP_DIR=%~2"
set "STEP_SCRIPT=%~3"
if not exist "%STEP_DIR%\%STEP_SCRIPT%" (
    echo Missing training entrypoint: %STEP_DIR%\%STEP_SCRIPT%
    exit /b 1
)

echo.
echo [%STEP_NAME%] "%PYTHON_EXE%" %STEP_SCRIPT%

if "%DRY_RUN%"=="1" (
    exit /b 0
)

pushd "%STEP_DIR%"
"%PYTHON_EXE%" "%STEP_SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo Step "%STEP_NAME%" failed with exit code %EXIT_CODE%.
    if "%CONTINUE_ON_ERROR%"=="1" (
        exit /b 0
    )
    exit /b %EXIT_CODE%
)

exit /b 0
