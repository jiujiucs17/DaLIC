# SPDX-License-Identifier: Apache-2.0
# Added in this derivative repository for the DaLIC x LocAgent experimental suite.

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
DEFAULT_OUTPUT_FILE = "results.xlsx"
DIFF_STATS_FILE = "diff_stats.json"
CONFIG_LABELS = {
    "original": "Baseline",
    "original_without_graph": "Baseline - Graph",
    "with_dalic": "DaLIC(as prompt, trace)",
    "with_dalic_without_graph": "DaLIC(as prompt, trace) - Graph",
    "with_dalic_data_deps": "DaLIC(as prompt, trace + data)",
    "with_dalic_data_deps_without_graph": "DaLIC(as prompt, trace + data) - Graph",
    "with_dalic_trace_tool_raw": "DaLIC(as tool, trace)",
    "with_dalic_trace_tool_raw_without_graph": "DaLIC(as tool, trace) - Graph",
    "with_dalic_trace_data_deps_tool_raw": "DaLIC(as tool, trace + data)",
    "with_dalic_trace_data_deps_tool_raw_without_graph": "DaLIC(as tool, trace + data) - Graph",
}


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
        if json_file.name == DIFF_STATS_FILE:
            continue
        config_names.append(json_file.stem)
    return config_names


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_config_summary(outputs_dir, config_name):
    return load_json(outputs_dir / f"{config_name}.json")


def get_config_label(config_name):
    return CONFIG_LABELS.get(config_name, config_name)


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
                    "Config Label": get_config_label(config_name),
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
            "Config Label",
            "Instance ID",
            "Run ID",
            "Level",
            "Precision",
            "Recall",
            "F1",
        ],
    )


def build_diff_stats_dataframe(outputs_dir):
    diff_stats = load_json(outputs_dir / DIFF_STATS_FILE)
    rows = []

    for pair_id in sorted(diff_stats):
        if pair_id == "_meta":
            continue

        pair_stats = diff_stats[pair_id]
        for section_name in ("performance_stats", "stability_stats"):
            section_stats = pair_stats[section_name]
            for metric, metric_stats in section_stats.items():
                if section_name == "performance_stats":
                    for level, stat_block in metric_stats.items():
                        rows.append(
                            {
                                "Config A Name": pair_stats["config_a_name"],
                                "Config A Label": get_config_label(pair_stats["config_a_name"]),
                                "Config B Name": pair_stats["config_b_name"],
                                "Config B Label": get_config_label(pair_stats["config_b_name"]),
                                "Section": section_name,
                                "Metric": metric,
                                "Aggregation Type": "AVG",
                                "Level": level,
                                "p": stat_block["p"],
                                "one_minus_p_percent": stat_block["one_minus_p_percent"],
                                "effect_size": stat_block["effect_size"],
                                "CI Low": stat_block["CI"][0],
                                "CI High": stat_block["CI"][1],
                            }
                        )
                else:
                    for stability_type, stability_stats in metric_stats.items():
                        for level, stat_block in stability_stats.items():
                            rows.append(
                                {
                                    "Config A Name": pair_stats["config_a_name"],
                                    "Config A Label": get_config_label(pair_stats["config_a_name"]),
                                    "Config B Name": pair_stats["config_b_name"],
                                    "Config B Label": get_config_label(pair_stats["config_b_name"]),
                                    "Section": section_name,
                                    "Metric": metric,
                                    "Aggregation Type": stability_type,
                                    "Level": level,
                                    "p": stat_block["p"],
                                    "one_minus_p_percent": stat_block["one_minus_p_percent"],
                                    "effect_size": stat_block["effect_size"],
                                    "CI Low": stat_block["CI"][0],
                                    "CI High": stat_block["CI"][1],
                                }
                            )

    return pd.DataFrame(
        rows,
        columns=[
            "Config A Name",
            "Config A Label",
            "Config B Name",
            "Config B Label",
            "Section",
            "Metric",
            "Aggregation Type",
            "Level",
            "p",
            "one_minus_p_percent",
            "effect_size",
            "CI Low",
            "CI High",
        ],
    )


def build_stability_raw_data_dataframe(outputs_dir):
    rows = []
    config_names = discover_config_names(outputs_dir)
    if not config_names:
        raise ValueError(f"No config json files found under {outputs_dir}")

    for config_name in config_names:
        summary = load_config_summary(outputs_dir, config_name)
        config_label = get_config_label(config_name)
        for instance_id in sorted(summary):
            for level in LEVEL_ORDER:
                level_summary = summary[instance_id][level]
                for metric in METRIC_NAMES:
                    rows.append(
                        {
                            "Config Name": config_name,
                            "Config Label": config_label,
                            "Instance ID": instance_id,
                            "Level": level,
                            "Metric": metric,
                            "SD": level_summary[metric]["standard_deviation"],
                            "MAD": level_summary[metric]["mean_avg_deviation"],
                        }
                    )

    return pd.DataFrame(
        rows,
        columns=[
            "Config Name",
            "Config Label",
            "Instance ID",
            "Level",
            "Metric",
            "SD",
            "MAD",
        ],
    )


def write_excel(outputs_dir, output_file, run_data_df, diff_stats_df, stability_raw_data_df):
    output_path = outputs_dir / output_file
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        run_data_df.to_excel(writer, sheet_name="run_data", index=False)
        diff_stats_df.to_excel(writer, sheet_name="diff_stats", index=False)
        stability_raw_data_df.to_excel(writer, sheet_name="stability raw data", index=False)
    return output_path


def main():
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    run_data_df = build_run_data_dataframe(outputs_dir)
    diff_stats_df = build_diff_stats_dataframe(outputs_dir)
    stability_raw_data_df = build_stability_raw_data_dataframe(outputs_dir)
    output_path = write_excel(
        outputs_dir, args.output_file, run_data_df, diff_stats_df, stability_raw_data_df
    )

    print(f"Found {run_data_df['Config Name'].nunique()} configs.")
    print(f"Flattened {len(diff_stats_df)} diff-stats rows.")
    print(f"Flattened {len(stability_raw_data_df)} stability raw-data rows.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
