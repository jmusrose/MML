param(
    [string]$PythonExe = "E:\anaconda3\envs\pytorch2.5\python.exe",
    [switch]$DryRun,
    [switch]$ContinueOnError,
    [int]$EarlyStopPatience = 3,
    [double]$EarlyStopMinDelta = 0.0,
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

$CommonArgs = @(
    "--early_stop_patience", [string]$EarlyStopPatience,
    "--early_stop_min_delta", [string]$EarlyStopMinDelta
)

$Steps = @(
    @{
        Name = "RGB_v1 NYU"
        WorkingDirectory = Join-Path $Root "RGB_v1"
        ScriptPath = "RGB_v1\DML_nyu.py"
        Script = "DML_nyu.py"
        Args = $CommonArgs + $RgbNyuArgs
    },
    @{
        Name = "RGB_v1 SUN"
        WorkingDirectory = Join-Path $Root "RGB_v1"
        ScriptPath = "RGB_v1\DML_sun.py"
        Script = "DML_sun.py"
        Args = $CommonArgs + $RgbSunArgs
    },
    @{
        Name = "MVSA_v1"
        WorkingDirectory = Join-Path $Root "MVSA_v1"
        ScriptPath = "MVSA_v1\DML_MVSA.py"
        Script = "DML_MVSA.py"
        Args = $CommonArgs + $MvsaArgs
    },
    @{
        Name = "Food_v1"
        WorkingDirectory = Join-Path $Root "Food_v1"
        ScriptPath = "Food_v1\DML_Food.py"
        Script = "DML_Food.py"
        Args = $CommonArgs + $FoodArgs
    },
    @{
        Name = "CREMAD_v1"
        WorkingDirectory = Join-Path $Root "CREMAD_v1"
        ScriptPath = "CREMAD_v1\DML_cremad.py"
        Script = "DML_cremad.py"
        Args = $CommonArgs + $CremadArgs
    }
)

foreach ($Step in $Steps) {
    $scriptFile = Join-Path $Root $Step.ScriptPath
    if (-not (Test-Path -LiteralPath $scriptFile)) {
        throw "Missing training entrypoint: $scriptFile"
    }

    $commandText = @($PythonExe, $Step.Script) + $Step.Args
    Write-Host ""
    Write-Host "[$($Step.Name)] $($commandText -join ' ')"

    if ($DryRun) {
        continue
    }

    Push-Location -LiteralPath $Step.WorkingDirectory
    try {
        & $PythonExe $Step.Script @($Step.Args)
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
Write-Host "All requested v1 project runs finished."
