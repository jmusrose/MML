from pathlib import Path


def test_run_all_projects_script_exists_and_runs_expected_order():
    script = Path("run_all_projects.ps1")
    assert script.exists()

    text = script.read_text(encoding="utf-8")
    expected = [
        "CREMAD_v1\\DML_cremad.py",
        "Food_v1\\DML_Food.py",
        "MVSA\\DML_MVSA.py",
        "RGB_v1\\DML_nyu.py",
        "RGB_v1\\DML_sun.py",
    ]

    positions = [text.index(item) for item in expected]
    assert positions == sorted(positions)


def test_run_all_projects_script_supports_dry_run_and_failure_policy():
    text = Path("run_all_projects.ps1").read_text(encoding="utf-8")

    assert "[switch]$DryRun" in text
    assert "[switch]$ContinueOnError" in text
    assert "$LASTEXITCODE" in text
    assert "pytorch2.5" in text


def test_run_all_projects_bat_exists_and_runs_expected_order():
    script = Path("run_all_projects.bat")
    assert script.exists()

    lines = [
        line.strip()
        for line in script.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("call :run_step")
    ]
    text = "\n".join(lines)
    expected = [
        "RGB_v1 NYU",
        "DML_nyu.py",
        "RGB_v1 SUN",
        "DML_sun.py",
        "MVSA_v1",
        "DML_MVSA.py",
        "Food_v1",
        "DML_Food.py",
        "CREMAD_v1",
        "DML_cremad.py",
    ]

    positions = [text.index(item) for item in expected]
    assert positions == sorted(positions)


def test_run_all_projects_bat_supports_dry_run_and_failure_policy():
    text = Path("run_all_projects.bat").read_text(encoding="utf-8")

    assert "--dry-run" in text
    assert "--continue-on-error" in text
    assert "pytorch2.5" in text
    assert "%ERRORLEVEL%" in text
