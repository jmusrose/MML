param(
    [string]$PythonExe = "E:\anaconda3\envs\pytorch2.5\python.exe",
    [switch]$DryRun,
    [switch]$ContinueOnError,
    [string]$RgbNyuIbBeta = "1e-2",
    [string]$IbBeta = "1e-3",
    [string]$IbEpsScale = "1.0",
    [string]$RgbNyuConformalAlpha = "0.05",
    [string]$ConformalAlpha = "0.1",
    [string]$UncertaintyTau = "1.0",
    [string]$CalibSize = "0",
    [string]$SunValSplitRatio = "0.2",
    [string[]]$RgbNyuArgs = @(),
    [string[]]$RgbSunArgs = @(),
    [string[]]$MvsaArgs = @(),
    [string[]]$FoodArgs = @(),
    [string[]]$CremadArgs = @()
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Warning "Python executable not found at '$PythonExe'; falling back to 'python' on PATH."
    $PythonExe = "python"
}

$Steps = @(
    @{
        Name = "RGB_v4 NYU"
        WorkingDirectory = Join-Path $Root "RGB_v4"
        ScriptPath = "RGB_v4\DML_nyu.py"
        Script = "DML_nyu.py"
        Args = @(
            "--ib_beta", $RgbNyuIbBeta,
            "--ib_eps_scale", $IbEpsScale,
            "--conformal_alpha", $RgbNyuConformalAlpha,
            "--uncertainty_tau", $UncertaintyTau
        ) + $RgbNyuArgs
    },
    @{
        Name = "RGB_v4 SUN"
        WorkingDirectory = Join-Path $Root "RGB_v4"
        ScriptPath = "RGB_v4\DML_sun.py"
        Script = "DML_sun.py"
        Args = @(
            "--ib_beta", $IbBeta,
            "--ib_eps_scale", $IbEpsScale,
            "--conformal_alpha", $ConformalAlpha,
            "--uncertainty_tau", $UncertaintyTau,
            "--val_split_ratio", $SunValSplitRatio
        ) + $RgbSunArgs
    },
    @{
        Name = "MVSA_v4"
        WorkingDirectory = Join-Path $Root "MVSA_v4"
        ScriptPath = "MVSA_v4\DML_MVSA.py"
        Script = "DML_MVSA.py"
        Args = @(
            "--ib_beta", $IbBeta,
            "--ib_eps_scale", $IbEpsScale,
            "--conformal_alpha", $ConformalAlpha,
            "--uncertainty_tau", $UncertaintyTau,
            "--calib_size", $CalibSize
        ) + $MvsaArgs
    },
    @{
        Name = "Food_v4"
        WorkingDirectory = Join-Path $Root "Food_v4"
        ScriptPath = "Food_v4\DML_Food.py"
        Script = "DML_Food.py"
        Args = @(
            "--ib_beta", $IbBeta,
            "--ib_eps_scale", $IbEpsScale,
            "--conformal_alpha", $ConformalAlpha,
            "--uncertainty_tau", $UncertaintyTau,
            "--calib_size", $CalibSize
        ) + $FoodArgs
    },
    @{
        Name = "CREMAD_v4"
        WorkingDirectory = Join-Path $Root "CREMAD_v4"
        ScriptPath = "CREMAD_v4\DML_cremad.py"
        Script = "DML_cremad.py"
        Args = @(
            "--config", "data\crema.json",
            "--conformal_alpha", $ConformalAlpha,
            "--uncertainty_tau", $UncertaintyTau,
            "--calib_size", $CalibSize
        ) + $CremadArgs
    }
)

foreach ($Step in $Steps) {
    $scriptFile = Join-Path $Root $Step.ScriptPath
    if (-not (Test-Path -LiteralPath $scriptFile)) {
        throw "Missing training entrypoint: $scriptFile"
    }

    $stepArgs = @($Step.Args)
    $commandText = @($PythonExe, $Step.Script) + $stepArgs
    Write-Host ""
    Write-Host "[$($Step.Name)] $($commandText -join ' ')"

    if ($DryRun) {
        continue
    }

    Push-Location -LiteralPath $Step.WorkingDirectory
    try {
        & $PythonExe $Step.Script @stepArgs
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    if ($exitCode -ne 0) {
        $message = "Step '$($Step.Name)' failed with exit code $exitCode."
        if ($ContinueOnError) {
            Write-Warning $message
            continue
        }
        throw $message
    }
}

Write-Host ""
Write-Host "All requested v4 information bottleneck and conformal project runs finished."
