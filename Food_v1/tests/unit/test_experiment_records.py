#!/usr/bin/env python3
"""Tests for RGB-style experiment summary persistence."""

import json

from utils.utils import append_experiment_record


def test_append_experiment_record_creates_json_array(tmp_path):
    summary_path = tmp_path / "all_experiments.json"
    record = {"dataset": "food101", "best_clean_acc": 0.75}

    append_experiment_record(str(summary_path), record)

    assert json.loads(summary_path.read_text(encoding="utf-8")) == [record]


def test_append_experiment_record_preserves_existing_records(tmp_path):
    summary_path = tmp_path / "all_experiments.json"
    first = {"dataset": "food101", "best_clean_acc": 0.70}
    second = {"dataset": "food101", "best_clean_acc": 0.80}
    summary_path.write_text(json.dumps([first]), encoding="utf-8")

    append_experiment_record(str(summary_path), second)

    assert json.loads(summary_path.read_text(encoding="utf-8")) == [first, second]
