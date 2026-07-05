"""
Aggregate experiment results into comparison tables.

Reads metrics.csv / summary.json from CNN and PINN result directories,
outputs comparison.csv and comparison.md.

Usage:
  python aggregate_results.py
  python aggregate_results.py --results-dir results/experiments/20260705_120000
"""

import argparse, csv, json, sys
from datetime import datetime
from pathlib import Path


def find_result_dirs(base):
    """Walk base directory and collect all subdirectories containing any *metrics*.csv."""
    dirs = []
    for metrics in sorted(Path(base).rglob("*metrics*.csv")):
        dirs.append((metrics.parent, metrics.name))
    return dirs


def read_metrics(result_dir, csv_name="metrics.csv"):
    csv_path = result_dir / csv_name
    summary_path = result_dir / "summary.json"
    if not csv_path.exists():
        return None

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return {"rows": rows, "summary": summary}


def classify(result_dir, csv_name="metrics.csv"):
    """Heuristic: determine experiment type from path and filename."""
    p = str(result_dir)
    exp = {"data_source": "unknown", "training_loss": "unknown",
           "test_type": "data", "method": "CNN"}

    # detect by csv filename first
    if "analytic" in csv_name:
        exp["test_type"] = "analytic"
    elif "fixed_solver" in csv_name:
        exp["test_type"] = "fixed-solver"

    if "pinn" in p.lower():
        exp["method"] = "PINN"
        exp["training_loss"] = "pinn"
        exp["data_source"] = "analytic"
    elif "mms_" in p or "mms/" in p:
        exp["data_source"] = "mms"
        if "_sup" in p:
            exp["training_loss"] = "supervised"
        elif "_physics" in p:
            exp["training_loss"] = "physics"
    elif "solver_" in p or "solver/" in p:
        exp["data_source"] = "solver"
        if "_sup" in p:
            exp["training_loss"] = "supervised"
        elif "_physics" in p:
            exp["training_loss"] = "physics"

    return exp


def build_table(all_dirs):
    rows = []
    for d, csv_name in sorted(all_dirs):
        info = classify(d, csv_name)
        data = read_metrics(d, csv_name)
        if data is None or not data["rows"]:
            continue

        summary = data["summary"]
        row = {
            "directory": f"{d}/{csv_name}",
            "experiment_name": d.parent.name if d.parent.name != "figures" else d.name,
            "data_source": info["data_source"],
            "training_loss": info["training_loss"],
            "test_type": info["test_type"],
            "method": info["method"],
            "n_samples": len(data["rows"]),
        }

        # extract k from first row
        row["k_test"] = data["rows"][0].get("k", "?")

        # extract key metrics from summary (mean values)
        for key in ["relL2", "maxError", "predResRMSE", "refResRMSE",
                     "bndErr", "gradRelL2", "gradLinf",
                     "constBaselineRelL2", "linearBaselineRelL2"]:
            if key in summary:
                row[f"mean_{key}"] = summary[key]["mean"]
            elif key in data["rows"][0]:
                row[f"mean_{key}"] = float(data["rows"][0][key])

        # time
        row["solve_time_sec"] = float(data["rows"][0].get("solve_time_sec", 0))

        rows.append(row)
    return rows


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows, path):
    if not rows:
        return

    float_keys = [k for k in rows[0] if k.startswith("mean_") or k in ("solve_time_sec",)]
    display_keys = ["experiment_name", "data_source", "training_loss",
                     "test_type", "method", "k_test", "n_samples"] + float_keys

    lines = [
        "# Experiment Comparison",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Methodology Notes",
        "",
        "- **CNN**: operator learning — trained once on a dataset of (f,g,k) pairs, "
        "then a single forward pass solves any new PDE instance. "
        "Training is computationally intensive but inference is near-instantaneous.",
        "- **PINN**: per-instance optimization — the neural network is optimized "
        "from scratch for each new PDE instance using the PDE residual as the loss. "
        "No training dataset is needed, but each new instance requires its own optimization.",
        "",
        "Because these are fundamentally different paradigms, "
        "**do not compare training/inference times directly**. "
        "Focus on accuracy metrics (relL2, physics residual) and boundary/constraint satisfaction.",
        "",
        "## K Test Note",
        "",
        "- Training uses random k ~ U(0,5) for CNN experiments.",
        "- Test (data mode) uses fixed --predict-k (default 2).",
        "- Analytic/fixed-solver test uses k=2 (required: signed u supports only integer k; "
        "chosen k=2 as the simplest non-trivial even power).",
        "",
        "## Results",
        "",
    ]

    header = "| " + " | ".join(display_keys) + " |"
    sep = "|" + "|".join(["---" for _ in display_keys]) + "|"
    lines.append(header)
    lines.append(sep)

    for row in rows:
        vals = []
        for k in display_keys:
            v = row.get(k, "")
            if isinstance(v, float):
                if k == "solve_time_sec":
                    vals.append(f"{v:.3f}")
                elif abs(v) < 1e-10:
                    vals.append("~0")
                else:
                    vals.append(f"{v:.4e}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  comparison.md → {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results",
                   help="Root directory to scan for experiment results")
    args = p.parse_args()

    base = Path(args.results_dir)
    if not base.exists():
        print(f"ERROR: {base} not found")
        sys.exit(1)

    dirs = find_result_dirs(base)
    if not dirs:
        print(f"No metrics.csv found under {base}")
        sys.exit(1)

    print(f"Found {len(dirs)} result directories")

    rows = build_table(dirs)
    if not rows:
        print("No parseable metrics found")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / "experiments" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "comparison.csv"
    write_csv(rows, csv_path)
    print(f"  comparison.csv → {csv_path}")

    md_path = out_dir / "comparison.md"
    write_markdown(rows, md_path)

    print(f"\nDone. {len(rows)} experiments aggregated.")


if __name__ == "__main__":
    main()
