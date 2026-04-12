import argparse
import json
from pathlib import Path

import pandas as pd


LEVEL_ORDER = ["function", "module", "file"]
METRIC_NAMES = {
    "precision": "Precision",
    "recall": "Recall",
    "F1": "F1",
}
DEFAULT_OUTPUT_FILE = "run_data.xlsx"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Path to DaLIC/outputs.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Excel file name to create under outputs-dir.",
    )
    return parser.parse_args()


def discover_config_names(outputs_dir):
    config_names = []
    for json_file in sorted(outputs_dir.glob("*.json")):
        if json_file.name == "diff_stats.json":
            continue
        config_names.append(json_file.stem)
    return config_names


def load_config_summary(outputs_dir, config_name):
    summary_path = outputs_dir / f"{config_name}.json"
    with open(summary_path, "r") as f:
        return json.load(f)


def build_run_rows(config_name, summary):
    rows = []
    for instance_id in sorted(summary):
        for level in LEVEL_ORDER:
            level_summary = summary[instance_id][level]
            metric_values = {
                metric: level_summary[metric]["raw_data"]
                for metric in METRIC_NAMES
            }
            run_count = max(len(values) for values in metric_values.values())

            for run_index in range(run_count):
                row = {
                    "Config Name": config_name,
                    "Instance ID": instance_id,
                    "Run ID": run_index + 1,
                    "Level": level,
                }
                for metric, column_name in METRIC_NAMES.items():
                    values = metric_values[metric]
                    row[column_name] = values[run_index] if run_index < len(values) else None
                rows.append(row)
    return rows


def build_run_data_dataframe(outputs_dir):
    rows = []
    config_names = discover_config_names(outputs_dir)
    if not config_names:
        raise ValueError(f"No config json files found under {outputs_dir}")

    for config_name in config_names:
        summary = load_config_summary(outputs_dir, config_name)
        rows.extend(build_run_rows(config_name, summary))

    return pd.DataFrame(
        rows,
        columns=[
            "Config Name",
            "Instance ID",
            "Run ID",
            "Level",
            "Precision",
            "Recall",
            "F1",
        ],
    )


def write_excel(outputs_dir, output_file, run_data_df):
    output_path = outputs_dir / output_file
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        run_data_df.to_excel(writer, sheet_name="run_data", index=False)
    return output_path


def main():
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    run_data_df = build_run_data_dataframe(outputs_dir)
    output_path = write_excel(outputs_dir, args.output_file, run_data_df)

    print(f"Found {run_data_df['Config Name'].nunique()} configs.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
