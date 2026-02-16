"""
Small helper to reproduce Table 5 (Concept Accuracy, Steerability, Perplexity)
by simply running the commands from the README via subprocess, similar to the
Table 2 automation script.

Usage inside Colab:
    import os
    os.chdir("/content/capstone/classification")
    !python reproduce_table5.py
"""

import re
import subprocess
from pathlib import Path

import pandas as pd


CLASSIFICATION_DIR = Path(__file__).resolve().parent
GENERATION_DIR = CLASSIFICATION_DIR.parent / "generation"

DATASETS = [
    ("SST2", "SetFit/sst2", 3),
    ("YelpP", "yelp_polarity", 1),
    ("AGnews", "ag_news", 1),
    ("DBpedia", "dbpedia_14", 1),
]

ACCURACY_PATTERN = re.compile(r"\{'accuracy':\s*([0-9.]+)\}")


def run(cmd: str) -> str:
    print(f"\n=== Running: {cmd} ===\n")
    completed = subprocess.run(
        cmd,
        cwd=GENERATION_DIR,
        shell=True,
        check=True,
        text=True,
        capture_output=True,
    )
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    return completed.stdout


def parse_accuracy(output: str) -> float:
    match = ACCURACY_PATTERN.search(output)
    if not match:
        raise ValueError("Could not parse accuracy from output.")
    return float(match.group(1))


def parse_perplexity(output: str) -> float:
    for line in reversed(output.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return float(stripped)
    raise ValueError("Could not parse perplexity from output.")


def main():
    if not GENERATION_DIR.exists():
        raise FileNotFoundError(f"Generation directory not found: {GENERATION_DIR}")

    table_rows = []
    for label, dataset_flag, epochs in DATASETS:
        print(f"\n##### Dataset: {label} #####")

        concept_out = run(f"python test_concepts.py --dataset {dataset_flag}")
        concept_acc = parse_accuracy(concept_out)

        run(f"python train_classifier.py --dataset {dataset_flag}")
        rename_cmd = (
            f"cp {dataset_flag.replace('/', '_')}_classifier_epoch_{epochs}.pt "
            f"{dataset_flag.replace('/', '_')}_classifier.pt"
        )
        run(rename_cmd)

        steer_out = run(f"python test_steerability.py --dataset {dataset_flag}")
        steer_acc = parse_accuracy(steer_out)

        ppl_out = run(f"python test_perplexity.py --dataset {dataset_flag}")
        ppl = parse_perplexity(ppl_out)

        table_rows.append(
            (
                label,
                concept_acc,
                steer_acc,
                ppl,
            )
        )

    df = pd.DataFrame(
        table_rows,
        columns=["Dataset", "Concept Detection Accuracy", "Steerability", "Perplexity"],
    ).set_index("Dataset")

    print("\n=== Reproduced Table 5 ===")
    print(df)


if __name__ == "__main__":
    main()
