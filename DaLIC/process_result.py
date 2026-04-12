import argparse
import json
import math
import re
import sys
from pathlib import Path

from datasets import Dataset, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.utils import load_jsonl


LEVEL2KEY = {
    "file": "found_files",
    "module": "found_modules",
    "function": "found_entities",
}
LEVEL_ORDER = ["function", "module", "file"]
METRIC_ORDER = ["recall", "precision"]
SUMMARY_METRIC_ORDER = [*METRIC_ORDER, "F1"]
RUN_DIR_PATTERN = re.compile(r"(.+)_run_(\d+)$")
DEFAULT_DATASET = "JJcs17/Loc-Bench-add_fixed_commit"
DEFAULT_SPLIT = "test"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Path to DaLIC/outputs.",
    )
    return parser.parse_args()


def discover_config_runs(outputs_dir):
    config_runs = {}
    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        match = RUN_DIR_PATTERN.fullmatch(child.name)
        if not match:
            continue
        config_name, run_id = match.group(1), int(match.group(2))
        config_runs.setdefault(config_name, {})[run_id] = child
    return config_runs


def load_run_args(config_runs):
    for run_dirs in config_runs.values():
        for run_dir in run_dirs.values():
            args_path = run_dir / "args.json"
            if args_path.exists():
                with open(args_path, "r") as f:
                    return json.load(f)
    return {}


def _normalize_function_name(function_name):
    if function_name.endswith(".__init__"):
        return function_name[: -len(".__init__")]
    return function_name


def calc_full_metric_from_lists(metric, pred_locs, gt_locs):
    gt_set = set(gt_locs)
    pred_hits = [1 if loc in gt_set else 0 for loc in pred_locs]

    if metric == "precision":
        return sum(pred_hits) / len(pred_hits) if pred_hits else 0.0
    if metric == "recall":
        return sum(pred_hits) / len(gt_locs) if gt_locs else 0.0
    raise ValueError(f"Unsupported metric: {metric}")


def load_cached_dataset_from_arrow(dataset_name, split):
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    target_dir_name = dataset_name.replace("/", "___").lower()

    matching_dirs = [
        child for child in cache_root.iterdir()
        if child.is_dir() and child.name.lower() == target_dir_name
    ]
    if not matching_dirs:
        raise FileNotFoundError(f"No cached dataset dir found for {dataset_name}")

    arrow_files = []
    for dataset_dir in matching_dirs:
        arrow_files.extend(dataset_dir.glob(f"**/*-{split}.arrow"))
    if not arrow_files:
        raise FileNotFoundError(
            f"No cached arrow file found for {dataset_name} split={split}"
        )

    arrow_path = max(arrow_files, key=lambda path: path.stat().st_mtime)
    print(f"Falling back to cached arrow file: {arrow_path}")
    return Dataset.from_file(str(arrow_path))


def load_dataset_split(dataset_name, split):
    try:
        return load_dataset(dataset_name, split=split)
    except PermissionError:
        return load_cached_dataset_from_arrow(dataset_name, split)


def build_ground_truth(dataset_name, split, instance_filter):
    dataset = load_dataset_split(dataset_name, split)
    instance_filter = set(instance_filter)
    gt_by_level = {level: {} for level in LEVEL_ORDER}

    for instance in dataset:
        instance_id = instance["instance_id"]
        if instance_filter and instance_id not in instance_filter:
            continue

        files = []
        modules = []
        functions = []

        for column in ("edit_functions", "added_functions"):
            for func in instance[column]:
                file_name = func.split(":")[0]
                module_name = func.split(":")[-1].split(".")[0]
                function_name = _normalize_function_name(func.split(":")[-1])

                if file_name not in files:
                    files.append(file_name)

                module_id = f"{file_name}:{module_name}"
                if module_id not in modules:
                    modules.append(module_id)

                function_id = f"{file_name}:{function_name}"
                if function_id not in functions:
                    functions.append(function_id)

        gt_by_level["file"][instance_id] = files
        gt_by_level["module"][instance_id] = modules
        gt_by_level["function"][instance_id] = functions

    return gt_by_level


def _normalize_pred_list(value):
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def load_predictions(loc_file):
    predictions = {level: {} for level in LEVEL_ORDER}
    for record in load_jsonl(loc_file):
        instance_id = record["instance_id"]
        predictions["file"][instance_id] = _normalize_pred_list(record.get(LEVEL2KEY["file"], []))
        predictions["module"][instance_id] = _normalize_pred_list(record.get(LEVEL2KEY["module"], []))
        predictions["function"][instance_id] = [
            _normalize_function_name(pred)
            for pred in _normalize_pred_list(record.get(LEVEL2KEY["function"], []))
        ]
    return predictions


def get_loc_file(run_dir):
    for file_name in ("loc_outputs.jsonl", "loc_output.jsonl"):
        loc_file = run_dir / file_name
        if loc_file.exists():
            return loc_file
    raise FileNotFoundError(f"Missing loc_outputs jsonl under {run_dir}")


def collect_instance_ids(config_runs):
    instance_ids = set()
    for run_dir in config_runs.values():
        for record in load_jsonl(get_loc_file(run_dir)):
            instance_ids.add(record["instance_id"])
    return sorted(instance_ids)


def score_run(gt_by_level, run_predictions, instance_ids):
    run_scores = {}
    for instance_id in instance_ids:
        run_scores[instance_id] = {}
        for level in LEVEL_ORDER:
            gt_locs = gt_by_level[level].get(instance_id, [])
            pred_locs = run_predictions[level].get(instance_id, [])
            run_scores[instance_id][level] = {}
            for metric in METRIC_ORDER:
                if not gt_locs:
                    score = 0.0
                else:
                    score = calc_full_metric_from_lists(metric, pred_locs, gt_locs)
                run_scores[instance_id][level][metric] = round(score, 4)
    return run_scores


def calc_standard_deviation(values):
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def calc_mean_avg_deviation(values):
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum(abs(value - mean_value) for value in values) / len(values)


def calc_f1_score(recall, precision):
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def summarize_config(config_name, run_dirs, gt_by_level, instance_ids):
    sorted_run_ids = sorted(run_dirs)
    summary = {
        instance_id: {
            level: {
                metric: {
                    "raw_data": [],
                    "standard_deviation": 0.0,
                    "mean_avg_deviation": 0.0,
                }
                for metric in SUMMARY_METRIC_ORDER
            }
            for level in LEVEL_ORDER
        }
        for instance_id in instance_ids
    }

    for run_id in sorted_run_ids:
        run_predictions = load_predictions(get_loc_file(run_dirs[run_id]))
        run_scores = score_run(gt_by_level, run_predictions, instance_ids)
        for instance_id in instance_ids:
            for level in LEVEL_ORDER:
                for metric in METRIC_ORDER:
                    summary[instance_id][level][metric]["raw_data"].append(
                        run_scores[instance_id][level][metric]
                    )
                recall = run_scores[instance_id][level]["recall"]
                precision = run_scores[instance_id][level]["precision"]
                summary[instance_id][level]["F1"]["raw_data"].append(
                    round(calc_f1_score(recall, precision), 4)
                )

    for instance_id in instance_ids:
        for level in LEVEL_ORDER:
            for metric in SUMMARY_METRIC_ORDER:
                raw_data = summary[instance_id][level][metric]["raw_data"]
                summary[instance_id][level][metric]["standard_deviation"] = round(
                    calc_standard_deviation(raw_data), 6
                )
                summary[instance_id][level][metric]["mean_avg_deviation"] = round(
                    calc_mean_avg_deviation(raw_data), 6
                )

    return summary


def write_summary(outputs_dir, config_name, summary):
    output_path = outputs_dir / f"{config_name}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return output_path


def main():
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    config_runs = discover_config_runs(outputs_dir)
    if not config_runs:
        raise ValueError(f"No config run folders found under {outputs_dir}")

    run_args = load_run_args(config_runs)
    dataset_name = run_args.get("dataset", DEFAULT_DATASET)
    split = run_args.get("split", DEFAULT_SPLIT)

    print(f"Using dataset: {dataset_name}")
    print(f"Using split: {split}")
    print("Ground-truth columns: edit_functions, added_functions")
    print(f"Found configs: {', '.join(sorted(config_runs))}")

    config_instance_ids = {
        config_name: collect_instance_ids(run_dirs)
        for config_name, run_dirs in sorted(config_runs.items())
    }
    all_instance_ids = sorted(
        {instance_id for instance_ids in config_instance_ids.values() for instance_id in instance_ids}
    )
    gt_by_level = build_ground_truth(dataset_name, split, all_instance_ids)

    for config_name, run_dirs in sorted(config_runs.items()):
        instance_ids = config_instance_ids[config_name]
        summary = summarize_config(config_name, run_dirs, gt_by_level, instance_ids)
        output_path = write_summary(outputs_dir, config_name, summary)
        print(
            f"Wrote {output_path.name} for {config_name} "
            f"with {len(run_dirs)} runs and {len(instance_ids)} instances."
        )


if __name__ == "__main__":
    main()
