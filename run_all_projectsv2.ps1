param(
    [string]$PythonExe = "E:\anaconda3\envs\pytorch2.5\python.exe",
    [switch]$DryRun,
    [switch]$ContinueOnError,
    [string]$IbBeta = "1e-3",
    [string]$IbEpsScale = "1.0",
    [string[]]$RgbNyuArgs = @(),
    [string[]]$RgbSunArgs = @(),
    [string[]]$MvsaArgs = @(),
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
        Name = "RGB_v2 NYU"
        WorkingDirectory = Join-Path $Root "RGB_v2"
        ScriptPath = "RGB_v2\DML_nyu.py"
        Script = "DML_nyu.py"
        Args = @("--ib_beta", $IbBeta, "--ib_eps_scale", $IbEpsScale) + $RgbNyuArgs
    },
    @{
        Name = "RGB_v2 SUN"
        WorkingDirectory = Join-Path $Root "RGB_v2"
        ScriptPath = "RGB_v2\DML_sun.py"
        Script = "DML_sun.py"
        Args = @("--ib_beta", $IbBeta, "--ib_eps_scale", $IbEpsScale) + $RgbSunArgs
    },
    @{
        Name = "MVSA_v2"
        WorkingDirectory = Join-Path $Root "MVSA_v2"
        ScriptPath = "MVSA_v2\DML_MVSA.py"
        Script = "DML_MVSA.py"
        Args = @("--ib_beta", $IbBeta, "--ib_eps_scale", $IbEpsScale) + $MvsaArgs
    },
    @{
        Name = "CREMAD_v2"
        WorkingDirectory = Join-Path $Root "CREMAD_v2"
        ScriptPath = "CREMAD_v2\DML_cremad.py"
        Script = "DML_cremad.py"
        Args = @("--config", "data\crema.json") + $CremadArgs
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
Write-Host "All requested v2 information bottleneck project runs finished."
