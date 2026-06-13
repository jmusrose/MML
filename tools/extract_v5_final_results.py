from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "实验结果"
OUT_XLSX = OUT_DIR / "v5_final_results_summary.xlsx"
OUT_WARMUP_CSV = OUT_DIR / "v5_warmup_final_results.csv"
OUT_V5_CSV = OUT_DIR / "v5_all_final_results.csv"

ALL_EXPERIMENT_FILES = [
    ROOT / "RGB_v5" / "savepath" / "nyud" / "all_experiments.json",
    ROOT / "RGB_v5" / "savepath" / "sun_rgbd" / "all_experiments.json",
    ROOT / "MVSA_v5" / "savepath" / "all_experiments.json",
    ROOT / "Food_v5" / "savepath" / "all_experiments.json",
    ROOT / "CREMAD_v5" / "savepath" / "CREMA-D" / "all_experiments.json",
    # CREMAD_v5 config currently writes final summaries here.
    ROOT / "CREMAD_v2" / "savepath" / "CREMA-D" / "all_experiments.json",
]

WARMUP_LOG_SUMMARY = ROOT / "v5_warmup_parameter_runs_extracted.csv"


ROBUSTNESS_COLUMNS = {
    "Clean Test": "clean_test",
    "Salt & Pepper (Lvl 5.0)": "salt_pepper_lvl5",
    "Salt & Pepper (Lvl 10.0)": "salt_pepper_lvl10",
    "Gaussian (Lvl 5.0)": "gaussian_lvl5",
    "Gaussian (Lvl 10.0)": "gaussian_lvl10",
}


BASE_COLUMNS = [
    "dataset",
    "experiment_name",
    "timestamp",
    "code_version_note",
    "source_file",
    "savedir",
    "note",
    "seed",
    "lr",
    "batch_sz",
    "max_epochs",
    "val_split_ratio",
    "ib_beta",
    "ib_eps_scale",
    "ib_warmup_epochs",
    "best_clean_epoch",
    "best_clean_acc",
    "clean_test",
    "salt_pepper_lvl5",
    "salt_pepper_lvl10",
    "gaussian_lvl5",
    "gaussian_lvl10",
    "param_summary",
    "remark",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def code_version_note(source_file: Path, savedir: str, has_warmup: bool) -> str:
    text = f"{source_file.as_posix()} {savedir}".lower()
    if "cremad_v2" in text and has_warmup:
        return "warmup_param; CREMAD_v5 run appears saved under CREMAD_v2 path"
    if "_v5" in text or "/v5" in text or "\\v5" in text:
        return "v5"
    if has_warmup:
        return "warmup_param"
    if "_v2" in text:
        return "v2/context"
    return "context"


def flatten_record(record: dict[str, Any], source_file: Path) -> dict[str, Any]:
    robustness = record.get("robustness") or {}
    savedir = str(record.get("savedir", ""))
    has_warmup = "ib_warmup_epochs" in record

    row: dict[str, Any] = {
        "dataset": record.get("dataset"),
        "experiment_name": record.get("name"),
        "timestamp": record.get("timestamp"),
        "code_version_note": code_version_note(source_file, savedir, has_warmup),
        "source_file": str(source_file.relative_to(ROOT)),
        "savedir": savedir,
        "note": record.get("note", ""),
        "seed": record.get("seed"),
        "lr": record.get("lr"),
        "batch_sz": record.get("batch_sz"),
        "max_epochs": record.get("max_epochs"),
        "val_split_ratio": record.get("val_split_ratio"),
        "ib_beta": record.get("ib_beta"),
        "ib_eps_scale": record.get("ib_eps_scale"),
        "ib_warmup_epochs": record.get("ib_warmup_epochs"),
        "best_clean_epoch": record.get("best_clean_epoch"),
        "best_clean_acc": record.get("best_clean_acc"),
    }

    for raw_name, col_name in ROBUSTNESS_COLUMNS.items():
        row[col_name] = robustness.get(raw_name)

    row["param_summary"] = (
        f"seed={row['seed']}; lr={row['lr']}; batch={row['batch_sz']}; "
        f"epochs={row['max_epochs']}; ib_beta={row['ib_beta']}; "
        f"ib_eps_scale={row['ib_eps_scale']}; warmup={row['ib_warmup_epochs']}"
    )
    remarks = []
    if has_warmup:
        remarks.append("has ib_warmup_epochs")
    if "CREMAD_v2" in savedir and has_warmup:
        remarks.append("savedir points to CREMAD_v2 although this is the warmup-param CREMAD run")
    if not robustness:
        remarks.append("missing robustness block")
    row["remark"] = "; ".join(remarks)
    return row


def collect_experiment_records() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in ALL_EXPERIMENT_FILES:
        if not path.exists():
            continue
        data = load_json(path)
        if isinstance(data, dict):
            data = [data]
        for record in data:
            if isinstance(record, dict):
                rows.append(flatten_record(record, path))

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[BASE_COLUMNS].sort_values(["dataset", "timestamp"], na_position="last")


def make_log_summary() -> pd.DataFrame:
    if not WARMUP_LOG_SUMMARY.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "source",
                "warmup_total",
                "rows",
                "warmup_best_val_epoch",
                "warmup_best_val",
                "first_kl_epoch",
                "first_kl_val",
                "first_kl_loss_ib",
                "last_logged_epoch",
                "last_logged_val",
                "remark",
            ]
        )

    raw = pd.read_csv(WARMUP_LOG_SUMMARY)
    raw["val_num"] = pd.to_numeric(raw["val_acc"], errors="coerce")
    raw["epoch_num"] = pd.to_numeric(raw["epoch"], errors="coerce")
    raw["loss_ib_num"] = pd.to_numeric(raw["loss_ib"], errors="coerce")

    rows = []
    for dataset, group in raw.groupby("dataset", sort=True):
        warmup_total = int(group["warmup_total"].dropna().iloc[0])
        warmup_rows = group[group["phase"] == "warmup"]
        kl_rows = group[group["phase"] == "kl_on"]

        warmup_best = warmup_rows.sort_values("val_num", ascending=False).head(1)
        first_kl = kl_rows.sort_values("epoch_num").head(1)
        last_logged = group.sort_values("epoch_num").tail(1)

        row = {
            "dataset": dataset,
            "source": group["source"].iloc[0],
            "warmup_total": warmup_total,
            "rows": len(group),
            "warmup_best_val_epoch": None,
            "warmup_best_val": None,
            "first_kl_epoch": None,
            "first_kl_val": None,
            "first_kl_loss_ib": None,
            "last_logged_epoch": None,
            "last_logged_val": None,
            "remark": "",
        }
        if not warmup_best.empty:
            row["warmup_best_val_epoch"] = int(warmup_best["epoch_num"].iloc[0])
            row["warmup_best_val"] = warmup_best["val_num"].iloc[0]
        if not first_kl.empty:
            row["first_kl_epoch"] = int(first_kl["epoch_num"].iloc[0])
            row["first_kl_val"] = first_kl["val_num"].iloc[0]
            row["first_kl_loss_ib"] = first_kl["loss_ib_num"].iloc[0]
        if not last_logged.empty:
            row["last_logged_epoch"] = int(last_logged["epoch_num"].iloc[0])
            row["last_logged_val"] = last_logged["val_num"].iloc[0]
        if dataset == "CREMAD":
            row["remark"] = "log summary is from pasted warmup=70 snippet; final test table also contains saved warmup=30 run"
        rows.append(row)

    return pd.DataFrame(rows)


def write_workbook(
    warmup_final: pd.DataFrame,
    v5_final: pd.DataFrame,
    all_records: pd.DataFrame,
    log_summary: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    field_notes = pd.DataFrame(
        [
            {
                "sheet": "warmup_final_results",
                "description": "Final robustness test rows with ib_warmup_epochs recorded.",
            },
            {
                "sheet": "v5_final_results_all",
                "description": "All records that look like v5-code runs or warmup-param runs.",
            },
            {
                "sheet": "all_experiment_records",
                "description": "All parsed records from the configured all_experiments.json files, including context baselines.",
            },
            {
                "sheet": "warmup_log_summary",
                "description": "Per-run summary parsed from v5_warmup_parameter_runs_extracted.csv.",
            },
            {
                "sheet": "field_notes",
                "description": "This sheet explains the workbook layout.",
            },
        ]
    )

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        warmup_final.to_excel(writer, sheet_name="warmup_final_results", index=False)
        v5_final.to_excel(writer, sheet_name="v5_final_results_all", index=False)
        all_records.to_excel(writer, sheet_name="all_experiment_records", index=False)
        log_summary.to_excel(writer, sheet_name="warmup_log_summary", index=False)
        field_notes.to_excel(writer, sheet_name="field_notes", index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column_cells in ws.columns:
                header = str(column_cells[0].value or "")
                width = max(len(header) + 2, 12)
                if header in {"source_file", "savedir", "param_summary", "remark", "description"}:
                    width = 48
                ws.column_dimensions[column_cells[0].column_letter].width = min(width, 80)


def main() -> None:
    all_records = collect_experiment_records()
    has_warmup = all_records["ib_warmup_epochs"].notna()
    looks_v5 = (
        all_records["savedir"].fillna("").str.contains("_v5|RGB_v5|MVSA_v5|Food_v5|CREMAD_v5", case=False, regex=True)
        | all_records["code_version_note"].fillna("").str.contains("warmup_param|v5", case=False, regex=True)
    )

    warmup_final = all_records[has_warmup].copy()
    v5_final = all_records[looks_v5].copy()
    log_summary = make_log_summary()

    warmup_final.to_csv(OUT_WARMUP_CSV, index=False, encoding="utf-8-sig")
    v5_final.to_csv(OUT_V5_CSV, index=False, encoding="utf-8-sig")
    write_workbook(warmup_final, v5_final, all_records, log_summary)

    print(f"Wrote {OUT_XLSX}")
    print(f"Wrote {OUT_WARMUP_CSV}")
    print(f"Wrote {OUT_V5_CSV}")
    print(f"warmup_final_rows={len(warmup_final)}")
    print(f"v5_final_rows={len(v5_final)}")
    print(f"all_records={len(all_records)}")


if __name__ == "__main__":
    main()
