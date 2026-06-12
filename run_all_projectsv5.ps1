param(
    [string]$PythonExe = "E:\anaconda3\envs\pytorch2.5\python.exe",
    [switch]$DryRun,
    [switch]$ContinueOnError,
    [string]$RgbNyuIbBeta = "1e-5",
    [string]$RgbSunIbBeta = "1e-4",
    [string]$IbBeta = "1e-3",
    [string]$IbEpsScale = "1.0",
    [string]$CremadIbEpsScale = "0.0",
    [string]$IbWarmupEpochs = "5",
    [string]$EarlyStopPatience = "3",
    [string]$EarlyStopMinDelta = "0.0",
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
        Name = "RGB_v5 NYU"
        WorkingDirectory = Join-Path $Root "RGB_v5"
        ScriptPath = "RGB_v5\DML_nyu.py"
        Script = "DML_nyu.py"
        Args = @(
            "--ib_beta", $RgbNyuIbBeta,
            "--ib_eps_scale", $IbEpsScale,
            "--ib_warmup_epochs", $IbWarmupEpochs,
            "--early_stop_patience", $EarlyStopPatience,
            "--early_stop_min_delta", $EarlyStopMinDelta
        ) + $RgbNyuArgs
    },
    @{
        Name = "RGB_v5 SUN"
        WorkingDirectory = Join-Path $Root "RGB_v5"
        ScriptPath = "RGB_v5\DML_sun.py"
        Script = "DML_sun.py"
        Args = @(
            "--ib_beta", $RgbSunIbBeta,
            "--ib_eps_scale", $IbEpsScale,
            "--ib_warmup_epochs", $IbWarmupEpochs,
            "--early_stop_patience", $EarlyStopPatience,
            "--early_stop_min_delta", $EarlyStopMinDelta,
            "--val_split_ratio", $SunValSplitRatio
        ) + $RgbSunArgs
    },
    @{
        Name = "MVSA_v5"
        WorkingDirectory = Join-Path $Root "MVSA_v5"
        ScriptPath = "MVSA_v5\DML_MVSA.py"
        Script = "DML_MVSA.py"
        Args = @(
            "--ib_beta", $IbBeta,
            "--ib_eps_scale", $IbEpsScale,
            "--ib_warmup_epochs", $IbWarmupEpochs,
            "--patience", $EarlyStopPatience
        ) + $MvsaArgs
    },
    @{
        Name = "Food_v5"
        WorkingDirectory = Join-Path $Root "Food_v5"
        ScriptPath = "Food_v5\DML_Food.py"
        Script = "DML_Food.py"
        Args = @(
            "--ib_beta", $IbBeta,
            "--ib_eps_scale", $IbEpsScale,
            "--ib_warmup_epochs", $IbWarmupEpochs,
            "--patience", $EarlyStopPatience
        ) + $FoodArgs
    },
    @{
        Name = "CREMAD_v5"
        WorkingDirectory = Join-Path $Root "CREMAD_v5"
        ScriptPath = "CREMAD_v5\DML_cremad.py"
        Script = "DML_cremad.py"
        Args = @(
            "--config", "data\crema.json",
            "--ib_beta", $IbBeta,
            "--ib_eps_scale", $CremadIbEpsScale,
            "--ib_warmup_epochs", $IbWarmupEpochs,
            "--early_stop_patience", $EarlyStopPatience,
            "--early_stop_min_delta", $EarlyStopMinDelta
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
Write-Host "All requested v5 feature information bottleneck project runs finished."
