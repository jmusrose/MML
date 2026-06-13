param(
    [string]$PythonExe = "E:\anaconda3\envs\pytorch2.5\python.exe",
    [switch]$DryRun,
    [switch]$ContinueOnError
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
    },
    @{
        Name = "RGB_v5 SUN"
        WorkingDirectory = Join-Path $Root "RGB_v5"
        ScriptPath = "RGB_v5\DML_sun.py"
        Script = "DML_sun.py"
    },
    @{
        Name = "MVSA_v5"
        WorkingDirectory = Join-Path $Root "MVSA_v5"
        ScriptPath = "MVSA_v5\DML_MVSA.py"
        Script = "DML_MVSA.py"
    },
    @{
        Name = "Food_v5"
        WorkingDirectory = Join-Path $Root "Food_v5"
        ScriptPath = "Food_v5\DML_Food.py"
        Script = "DML_Food.py"
    },
    @{
        Name = "CREMAD_v5"
        WorkingDirectory = Join-Path $Root "CREMAD_v5"
        ScriptPath = "CREMAD_v5\DML_cremad.py"
        Script = "DML_cremad.py"
    }
)

foreach ($Step in $Steps) {
    $scriptFile = Join-Path $Root $Step.ScriptPath
    if (-not (Test-Path -LiteralPath $scriptFile)) {
        throw "Missing training entrypoint: $scriptFile"
    }

    $commandText = @($PythonExe, $Step.Script)
    Write-Host ""
    Write-Host "[$($Step.Name)] $($commandText -join ' ')"

    if ($DryRun) {
        continue
    }

    Push-Location -LiteralPath $Step.WorkingDirectory
    try {
        & $PythonExe $Step.Script
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
