import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEVEL_ORDER = ["function", "module", "file"]
METRIC_ORDER = ["recall", "precision"]
LEVEL_GROUPS = {
    "function": ["function"],
    "module": ["module"],
    "file": ["file"],
    "3_levels": LEVEL_ORDER,
}
STABILITY_FIELD_MAP = {
    "SD": "standard_deviation",
    "MAD": "mean_avg_deviation",
}
NUM_RESAMPLES = 20000
RANDOM_SEED = 42
BATCH_SIZE = 5000


def build_metadata():
    return {
        "num_resamples": NUM_RESAMPLES,
        "effect_size_direction": "effect_size is mean(config_b - config_a). Positive means config_b performs better or is less variable, depending on the section.",
        "p_explanation": "p is the two-sided permutation-test p-value under the null hypothesis of no real difference.",
        "one_minus_p_percent_explanation": "one_minus_p_percent is (1 - p) * 100. It is a descriptive transformation of p.",
        "performance_stats_explanation": "performance_stats uses the mean of raw_data over the five runs.",
        "stability_stats_explanation": "stability_stats uses per-instance SD or MAD computed from the five raw_data values.",
        "sd_explanation": "SD is the per-instance standard deviation across the five runs.",
        "mad_explanation": "MAD is the per-instance mean absolute deviation from the five-run mean.",
        "3_levels_explanation": "3_levels pools function, module, and file samples together.",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Path to DaLIC/outputs.",
    )
    return parser.parse_args()


def load_config_summary(outputs_dir, config_name):
    summary_path = outputs_dir / f"{config_name}.json"
    with open(summary_path, "r") as f:
        return json.load(f)


def discover_config_names(outputs_dir):
    config_names = []
    for json_file in sorted(outputs_dir.glob("*.json")):
        if json_file.name == "diff_stats.json":
            continue
        config_names.append(json_file.stem)
    return config_names


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def collect_performance_samples(summary, metric, levels):
    samples = []
    for instance_id in sorted(summary):
        for level in levels:
            raw_data = summary[instance_id][level][metric]["raw_data"]
            samples.append(mean(raw_data))
    return np.asarray(samples, dtype=np.float64)


def collect_stability_samples(summary, metric, stability_name, levels):
    field_name = STABILITY_FIELD_MAP[stability_name]
    samples = []
    for instance_id in sorted(summary):
        for level in levels:
            samples.append(summary[instance_id][level][metric][field_name])
    return np.asarray(samples, dtype=np.float64)


def paired_permutation_test(differences, rng):
    observed = differences.mean()
    extreme_count = 0

    for start in range(0, NUM_RESAMPLES, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, NUM_RESAMPLES - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch_size, differences.size))
        permuted_means = (signs * differences).mean(axis=1)
        extreme_count += np.count_nonzero(np.abs(permuted_means) >= abs(observed))

    p_value = (extreme_count + 1) / (NUM_RESAMPLES + 1)
    return observed, p_value


def paired_bootstrap(differences, rng):
    bootstrap_means = []

    for start in range(0, NUM_RESAMPLES, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, NUM_RESAMPLES - start)
        indices = rng.integers(0, differences.size, size=(batch_size, differences.size))
        bootstrap_means.append(differences[indices].mean(axis=1))

    bootstrap_means = np.concatenate(bootstrap_means)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return differences.mean(), [float(ci_low), float(ci_high)]


def build_stat_block(samples_a, samples_b, seed_offset):
    if samples_a.shape != samples_b.shape:
        raise ValueError(f"Shape mismatch: {samples_a.shape} vs {samples_b.shape}")

    differences = samples_b - samples_a
    rng_perm = np.random.default_rng(RANDOM_SEED + seed_offset * 2)
    rng_boot = np.random.default_rng(RANDOM_SEED + seed_offset * 2 + 1)

    _, p_value = paired_permutation_test(differences, rng_perm)
    effect_size, ci_95 = paired_bootstrap(differences, rng_boot)
    return {
        "p": round(float(p_value), 6),
        "one_minus_p_percent": round(float((1.0 - p_value) * 100.0), 4),
        "effect_size": round(float(effect_size), 6),
        "CI": [round(float(ci_95[0]), 6), round(float(ci_95[1]), 6)],
    }


def build_performance_stats(summary_a, summary_b):
    performance_stats = {}
    seed_offset = 0
    for metric in METRIC_ORDER:
        performance_stats[metric] = {}
        for level_name, levels in LEVEL_GROUPS.items():
            samples_a = collect_performance_samples(summary_a, metric, levels)
            samples_b = collect_performance_samples(summary_b, metric, levels)
            performance_stats[metric][level_name] = build_stat_block(
                samples_a, samples_b, seed_offset
            )
            seed_offset += 1
    return performance_stats


def build_stability_stats(summary_a, summary_b):
    stability_stats = {}
    seed_offset = len(METRIC_ORDER) * len(LEVEL_GROUPS)
    for metric in METRIC_ORDER:
        stability_stats[metric] = {}
        for stability_name in ("MAD", "SD"):
            stability_stats[metric][stability_name] = {}
            for level_name, levels in LEVEL_GROUPS.items():
                samples_a = collect_stability_samples(summary_a, metric, stability_name, levels)
                samples_b = collect_stability_samples(summary_b, metric, stability_name, levels)
                stability_stats[metric][stability_name][level_name] = build_stat_block(
                    samples_a, samples_b, seed_offset
                )
                seed_offset += 1
    return stability_stats


def build_pair_stats(config_a_name, config_b_name, outputs_dir):
    summary_a = load_config_summary(outputs_dir, config_a_name)
    summary_b = load_config_summary(outputs_dir, config_b_name)

    return {
        "config_a_name": config_a_name,
        "config_b_name": config_b_name,
        "performance_stats": build_performance_stats(summary_a, summary_b),
        "stability_stats": build_stability_stats(summary_a, summary_b),
    }


def main():
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    config_names = discover_config_names(outputs_dir)
    if len(config_names) < 2:
        raise ValueError(f"Need at least two configs under {outputs_dir}")

    config_pairs = list(combinations(config_names, 2))
    results = {"_meta": build_metadata()}

    for config_a_name, config_b_name in tqdm(
        config_pairs, desc="Config pairs", total=len(config_pairs)
    ):
        pair_id = f"{config_a_name}__vs__{config_b_name}"
        results[pair_id] = build_pair_stats(config_a_name, config_b_name, outputs_dir)

    output_path = outputs_dir / "diff_stats.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Using {NUM_RESAMPLES} resamples for permutation test and paired bootstrap.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
