from pathlib import Path


def test_run_all_projects_script_exists_and_runs_expected_order():
    script = Path("run_all_projects.ps1")
    assert script.exists()

    text = script.read_text(encoding="utf-8")
    expected = [
        "RGB_v1\\DML_nyu.py",
        "RGB_v1\\DML_sun.py",
        "MVSA_v1\\DML_MVSA.py",
        "Food_v1\\DML_Food.py",
        "CREMAD_v1\\DML_cremad.py",
    ]

    positions = [text.index(item) for item in expected]
    assert positions == sorted(positions)
    assert "MVSA\\DML_MVSA.py" not in text


def test_run_all_projects_script_supports_dry_run_and_failure_policy():
    text = Path("run_all_projects.ps1").read_text(encoding="utf-8")

    assert "[switch]$DryRun" in text
    assert "[switch]$ContinueOnError" in text
    assert "$EarlyStopPatience" in text
    assert "$EarlyStopMinDelta" in text
    assert "--early_stop_patience" in text
    assert "--early_stop_min_delta" in text
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
    assert "--early-stop-patience" in text
    assert "--early-stop-min-delta" in text
    assert "--early_stop_patience" in text
    assert "--early_stop_min_delta" in text
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
        "Food_v2",
        "DML_Food.py",
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
    assert "$FoodArgs" in ps1_text
    assert "$EarlyStopPatience" in ps1_text
    assert "$EarlyStopMinDelta" in ps1_text
    assert "--early-stop-patience" in bat_text
    assert "--early-stop-min-delta" in bat_text
    assert "--early_stop_patience" in ps1_text
    assert "--early_stop_min_delta" in ps1_text
    assert "--patience" in ps1_text
    assert "$SunValSplitRatio" in ps1_text
    assert "--sun-val-split-ratio" in bat_text
    assert "--val_split_ratio" in ps1_text
    assert "--val_split_ratio" in bat_text
    assert "pytorch2.5" in ps1_text
    assert "pytorch2.5" in bat_text


def test_run_all_projectsv2_scripts_pass_expected_project_specific_args():
    ps1_text = Path("run_all_projectsv2.ps1").read_text(encoding="utf-8")
    bat_text = Path("run_all_projectsv2.bat").read_text(encoding="utf-8")

    assert "RGB_v2\\DML_nyu.py" in ps1_text
    assert "RGB_v2\\DML_sun.py" in ps1_text
    assert "MVSA_v2\\DML_MVSA.py" in ps1_text
    assert "Food_v2\\DML_Food.py" in ps1_text
    assert "CREMAD_v2\\DML_cremad.py" in ps1_text

    assert 'Script = "DML_sun.py"' in ps1_text
    assert '"--val_split_ratio", $SunValSplitRatio' in ps1_text
    assert 'Script = "DML_MVSA.py"' in ps1_text
    assert '"--patience", $EarlyStopPatience' in ps1_text
    assert 'Script = "DML_cremad.py"' in ps1_text
    assert '"--config", "data\\crema.json"' in ps1_text

    assert 'call :run_step "RGB_v2 SUN"' in bat_text
    assert '--val_split_ratio "%SUN_VAL_SPLIT_RATIO%"' in bat_text
    assert 'call :run_step "MVSA_v2"' in bat_text
    assert '--patience "%EARLY_STOP_PATIENCE%"' in bat_text
    assert 'call :run_step "CREMAD_v2"' in bat_text
    assert '--config "data\\crema.json"' in bat_text


def test_readme_documents_v2_information_bottleneck_runner():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "Food_v2" in text
    assert "CREMAD_v2" in text
    assert "run_all_projectsv2.bat" in text
    assert "run_all_projectsv2.ps1" in text
    assert "--ib-beta" in text
    assert "Information Bottleneck" in text


def test_run_all_projectsv4_scripts_exist_and_run_conformal_projects():
    ps1 = Path("run_all_projectsv4.ps1")
    bat = Path("run_all_projectsv4.bat")
    assert ps1.exists()
    assert bat.exists()

    ps1_text = ps1.read_text(encoding="utf-8")
    bat_text = bat.read_text(encoding="utf-8")
    expected = [
        "RGB_v4",
        "DML_nyu.py",
        "DML_sun.py",
        "MVSA_v4",
        "DML_MVSA.py",
        "Food_v4",
        "DML_Food.py",
        "CREMAD_v4",
        "DML_cremad.py",
    ]

    for item in expected:
        assert item in ps1_text
        assert item in bat_text

    assert "ib_beta" in ps1_text
    assert "ib_eps_scale" in ps1_text
    assert "conformal_alpha" in ps1_text
    assert "uncertainty_tau" in ps1_text
    assert "--ib-beta" not in bat_text
    assert "--ib-eps-scale" not in bat_text
    assert "--conformal-alpha" not in bat_text
    assert "--uncertainty-tau" not in bat_text
    assert "$CalibSize" in ps1_text
    assert "--calib-size" not in bat_text
    assert "$SunValSplitRatio" in ps1_text
    assert "--sun-val-split-ratio" not in bat_text
    assert "pytorch2.5" in ps1_text
    assert "pytorch2.5" in bat_text
    assert "run_all_projectsv4.ps1" not in bat_text


def test_run_all_projectsv4_scripts_pass_expected_project_specific_args():
    ps1_text = Path("run_all_projectsv4.ps1").read_text(encoding="utf-8")
    bat_text = Path("run_all_projectsv4.bat").read_text(encoding="utf-8")

    assert "RGB_v4\\DML_nyu.py" in ps1_text
    assert "RGB_v4\\DML_sun.py" in ps1_text
    assert "MVSA_v4\\DML_MVSA.py" in ps1_text
    assert "Food_v4\\DML_Food.py" in ps1_text
    assert "CREMAD_v4\\DML_cremad.py" in ps1_text

    assert 'Script = "DML_nyu.py"' in ps1_text
    assert '"--ib_beta", $RgbNyuIbBeta' in ps1_text
    assert '"--conformal_alpha", $RgbNyuConformalAlpha' in ps1_text
    assert 'Script = "DML_sun.py"' in ps1_text
    assert '"--val_split_ratio", $SunValSplitRatio' in ps1_text
    assert 'Script = "DML_MVSA.py"' in ps1_text
    assert 'Script = "DML_Food.py"' in ps1_text
    assert '"--calib_size", $CalibSize' in ps1_text
    assert 'Script = "DML_cremad.py"' in ps1_text
    assert '"--config", "data\\crema.json"' in ps1_text
    assert '"--patience"' not in ps1_text

    assert 'call :run_step "RGB_v4 NYU"' in bat_text
    assert 'call :run_step "RGB_v4 SUN"' in bat_text
    assert 'call :run_step "MVSA_v4"' in bat_text
    assert 'call :run_step "Food_v4"' in bat_text
    assert 'call :run_step "CREMAD_v4"' in bat_text
    assert "--ib_beta" not in bat_text
    assert "--conformal_alpha" not in bat_text
    assert "--uncertainty_tau" not in bat_text
    assert "--calib_size" not in bat_text
    assert "--val_split_ratio" not in bat_text
    assert "--config" not in bat_text
    assert '--patience' not in bat_text
