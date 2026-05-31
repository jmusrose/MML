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


def test_run_all_projectsv2_scripts_exist_and_run_information_bottleneck_projects():
    ps1 = Path("run_all_projectsv2.ps1")
    bat = Path("run_all_projectsv2.bat")
    assert ps1.exists()
    assert bat.exists()

    ps1_text = ps1.read_text(encoding="utf-8")
    bat_text = bat.read_text(encoding="utf-8")
    expected = [
        "RGB_v2",
        "DML_nyu.py",
        "DML_sun.py",
        "MVSA_v2",
        "DML_MVSA.py",
        "CREMAD_v2",
        "DML_cremad.py",
    ]

    for item in expected:
        assert item in ps1_text
        assert item in bat_text

    assert "ib_beta" in ps1_text
    assert "ib_eps_scale" in ps1_text
    assert "--ib-beta" in bat_text
    assert "--ib-eps-scale" in bat_text
    assert "pytorch2.5" in ps1_text
    assert "pytorch2.5" in bat_text


def test_readme_documents_v2_information_bottleneck_runner():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "CREMAD_v2" in text
    assert "run_all_projectsv2.bat" in text
    assert "run_all_projectsv2.ps1" in text
    assert "--ib-beta" in text
    assert "Information Bottleneck" in text
