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
echo Usage: run_rgb_v4_nyu_sun_ablation.bat [--dry-run] [--continue-on-error] [--python path\to\python.exe]
exit /b 2

:after_parse
if not exist "%PYTHON_EXE%" (
    echo Python executable not found: %PYTHON_EXE%
    echo Falling back to python on PATH.
    set "PYTHON_EXE=python"
)

set "STEP_ARGS=--ib_beta 0 --ib_eps_scale 0 --conformal_alpha 0.2 --savedir savepath\nyud_ablation\A_ib0_eps0_alpha0p2"
call :run_step "RGB_v4 NYU A" "%ROOT%RGB_v4" "DML_nyu.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 1e-5 --ib_eps_scale 0 --conformal_alpha 0.2 --savedir savepath\nyud_ablation\B_ib1e-5_eps0_alpha0p2"
call :run_step "RGB_v4 NYU B" "%ROOT%RGB_v4" "DML_nyu.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 1e-4 --ib_eps_scale 0 --conformal_alpha 0.2 --savedir savepath\nyud_ablation\C_ib1e-4_eps0_alpha0p2"
call :run_step "RGB_v4 NYU C" "%ROOT%RGB_v4" "DML_nyu.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 1e-4 --ib_eps_scale 0.1 --conformal_alpha 0.2 --savedir savepath\nyud_ablation\D_ib1e-4_eps0p1_alpha0p2"
call :run_step "RGB_v4 NYU D" "%ROOT%RGB_v4" "DML_nyu.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 0 --ib_eps_scale 0 --conformal_alpha 0.1 --savedir savepath\sun_rgbd_ablation\A_ib0_eps0_alpha0p1"
call :run_step "RGB_v4 SUN A" "%ROOT%RGB_v4" "DML_sun.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 1e-5 --ib_eps_scale 0 --conformal_alpha 0.1 --savedir savepath\sun_rgbd_ablation\B_ib1e-5_eps0_alpha0p1"
call :run_step "RGB_v4 SUN B" "%ROOT%RGB_v4" "DML_sun.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 1e-4 --ib_eps_scale 0 --conformal_alpha 0.1 --savedir savepath\sun_rgbd_ablation\C_ib1e-4_eps0_alpha0p1"
call :run_step "RGB_v4 SUN C" "%ROOT%RGB_v4" "DML_sun.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 1e-4 --ib_eps_scale 0.1 --conformal_alpha 0.1 --savedir savepath\sun_rgbd_ablation\D_ib1e-4_eps0p1_alpha0p1"
call :run_step "RGB_v4 SUN D" "%ROOT%RGB_v4" "DML_sun.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 0 --ib_eps_scale 0 --conformal_alpha 0.1 --savedir savepath\dml_mvsa_ablation\A_ib0_eps0_alpha0p1"
call :run_step "MVSA_v4 A" "%ROOT%MVSA_v4" "DML_MVSA.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 1e-6 --ib_eps_scale 0 --conformal_alpha 0.1 --savedir savepath\dml_mvsa_ablation\B_ib1e-6_eps0_alpha0p1"
call :run_step "MVSA_v4 B" "%ROOT%MVSA_v4" "DML_MVSA.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 1e-5 --ib_eps_scale 0 --conformal_alpha 0.1 --savedir savepath\dml_mvsa_ablation\C_ib1e-5_eps0_alpha0p1"
call :run_step "MVSA_v4 C" "%ROOT%MVSA_v4" "DML_MVSA.py"
if errorlevel 1 exit /b %errorlevel%

set "STEP_ARGS=--ib_beta 1e-5 --ib_eps_scale 0.1 --conformal_alpha 0.1 --savedir savepath\dml_mvsa_ablation\D_ib1e-5_eps0p1_alpha0p1"
call :run_step "MVSA_v4 D" "%ROOT%MVSA_v4" "DML_MVSA.py"
if errorlevel 1 exit /b %errorlevel%

echo.
echo RGB_v4 NYU/SUN and MVSA_v4 ablation runs finished.
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
echo [%STEP_NAME%] "%PYTHON_EXE%" %STEP_SCRIPT% %STEP_ARGS%

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
