@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "PYTHON_EXE=E:\anaconda3\envs\pytorch2.5\python.exe"
set "DRY_RUN=0"
set "CONTINUE_ON_ERROR=0"
set "IB_BETA=1e-3"
set "IB_EPS_SCALE=1.0"

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
if /I "%~1"=="--ib-beta" (
    set "IB_BETA=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--ib-eps-scale" (
    set "IB_EPS_SCALE=%~2"
    shift
    shift
    goto parse_args
)
echo Unknown argument: %~1
echo Usage: run_all_projectsv2.bat [--dry-run] [--continue-on-error] [--python path\to\python.exe] [--ib-beta value] [--ib-eps-scale value]
exit /b 2

:after_parse
if not exist "%PYTHON_EXE%" (
    echo Python executable not found: %PYTHON_EXE%
    echo Falling back to python on PATH.
    set "PYTHON_EXE=python"
)

call :run_step "RGB_v2 NYU" "%ROOT%RGB_v2" "DML_nyu.py" --ib_beta "%IB_BETA%" --ib_eps_scale "%IB_EPS_SCALE%"
if errorlevel 1 exit /b %errorlevel%

call :run_step "RGB_v2 SUN" "%ROOT%RGB_v2" "DML_sun.py" --ib_beta "%IB_BETA%" --ib_eps_scale "%IB_EPS_SCALE%"
if errorlevel 1 exit /b %errorlevel%

call :run_step "MVSA_v2" "%ROOT%MVSA_v2" "DML_MVSA.py" --ib_beta "%IB_BETA%" --ib_eps_scale "%IB_EPS_SCALE%"
if errorlevel 1 exit /b %errorlevel%

call :run_step "CREMAD_v2" "%ROOT%CREMAD_v2" "DML_cremad.py" --config "data\crema.json"
if errorlevel 1 exit /b %errorlevel%

echo.
echo All requested v2 information bottleneck project runs finished.
exit /b 0

:run_step
set "STEP_NAME=%~1"
set "STEP_DIR=%~2"
set "STEP_SCRIPT=%~3"
shift
shift
shift

set "STEP_ARGS="
:collect_step_args
if "%~1"=="" goto after_step_args
set "STEP_ARGS=!STEP_ARGS! "%~1""
shift
goto collect_step_args

:after_step_args
if not exist "%STEP_DIR%\%STEP_SCRIPT%" (
    echo Missing training entrypoint: %STEP_DIR%\%STEP_SCRIPT%
    exit /b 1
)

echo.
echo [%STEP_NAME%] "%PYTHON_EXE%" %STEP_SCRIPT%!STEP_ARGS!

if "%DRY_RUN%"=="1" (
    exit /b 0
)

pushd "%STEP_DIR%"
"%PYTHON_EXE%" "%STEP_SCRIPT%" %STEP_ARGS%
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
