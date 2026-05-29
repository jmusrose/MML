param(
    [string]$PythonExe = "E:\anaconda3\envs\pytorch2.5\python.exe",
    [switch]$DryRun,
    [switch]$ContinueOnError,
    [string[]]$CremadArgs = @(),
    [string[]]$FoodArgs = @(),
    [string[]]$MvsaArgs = @(),
    [string[]]$RgbNyuArgs = @(),
    [string[]]$RgbSunArgs = @()
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Warning "Python executable not found at '$PythonExe'; falling back to 'python' on PATH."
    $PythonExe = "python"
}

$Steps = @(
    @{
        Name = "CREMAD_v1"
        WorkingDirectory = Join-Path $Root "CREMAD_v1"
        ScriptPath = "CREMAD_v1\DML_cremad.py"
        Script = "DML_cremad.py"
        Args = $CremadArgs
    },
    @{
        Name = "Food_v1"
        WorkingDirectory = Join-Path $Root "Food_v1"
        ScriptPath = "Food_v1\DML_Food.py"
        Script = "DML_Food.py"
        Args = $FoodArgs
    },
    @{
        Name = "MVSA"
        WorkingDirectory = Join-Path $Root "MVSA"
        ScriptPath = "MVSA\DML_MVSA.py"
        Script = "DML_MVSA.py"
        Args = $MvsaArgs
    },
    @{
        Name = "RGB_v1 NYU"
        WorkingDirectory = Join-Path $Root "RGB_v1"
        ScriptPath = "RGB_v1\DML_nyu.py"
        Script = "DML_nyu.py"
        Args = $RgbNyuArgs
    },
    @{
        Name = "RGB_v1 SUN"
        WorkingDirectory = Join-Path $Root "RGB_v1"
        ScriptPath = "RGB_v1\DML_sun.py"
        Script = "DML_sun.py"
        Args = $RgbSunArgs
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
Write-Host "All requested project runs finished."
