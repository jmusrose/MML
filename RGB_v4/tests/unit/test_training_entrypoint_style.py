#!/usr/bin/env python3
"""Static checks for RGB v4 training entrypoint behavior."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_rgb_entrypoints_expose_conformal_uncertainty_tau():
    for script_name in ("DML_nyu.py", "DML_sun.py"):
        source = (PROJECT_ROOT / script_name).read_text(encoding="utf-8")

        assert "--uncertainty_tau" in source
        assert '"tau": float(args.uncertainty_tau)' in source
        assert "tau=args.uncertainty_tau" in source
