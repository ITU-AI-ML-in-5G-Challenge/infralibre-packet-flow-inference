#!/usr/bin/env python3
"""Train or run the public-data reference; no historical predictions are inputs."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from experiment_v4 import LABELS, SEEDS, aligned, constrained, engineer_single, model_set

STATIC = "task1_static"
PREFIX = "task2_realtime_prefix5"
RAW = [
    "start_rel", "duration", "transport", "server_port", "packet_count_up",
    "packet_count_down", "bytes_up", "bytes_down", "mean_pkt_size_up",
    "mean_pkt_size_down", "std_pkt_size_up", "std_pkt_size_down", "mean_iat",
    "std_iat", "min_iat", "max_iat", "burst_count", "up_down_packet_ratio",
    "up_down_byte_ratio",
]


def features(data, task, split):
    x = pd.read_csv(data / task / f"{split}_features.csv", usecols=["flow_id", *RAW],
                    dtype={"flow_id": str}).set_index("flow_id", verify_integrity=True)
    return engineer_single(x[RAW])


def labels(data, task, split, index):
    y = pd.read_csv(data / task / f"{split}_labels.csv", usecols=["flow_id", "app"],
                    dtype={"flow_id": str}).set_index("flow_id", verify_integrity=True).app
    if set(y.index) != set(index) or not set(y).issubset(set(LABELS)):
        raise ValueError(f"Invalid labels for {task}/{split}")
    return y.reindex(index).to_numpy(str)


def fit(data, task, splits):
    xx = [features(data, task, split) for split in splits]
    yy = [labels(data, task, split, x.index) for split, x in zip(splits, xx)]
    x, y = pd.concat(xx, verify_integrity=True), np.concatenate(yy)
    members = []
    for seed in SEEDS:
        models, weights = model_set("prefix" if task == PREFIX else "full", seed)
        for model, weight in zip(models, weights):
            keys = model.get_params()
            if "n_jobs" in keys:
                model.set_params(n_jobs=1)
            if "thread_count" in keys:
                model.set_params(thread_count=1)
            model.fit(x, y)
            members.append((model, weight / len(SEEDS)))
    return members


def count_assign(data, task, split, proba):
    counts = json.loads((data / task / "metadata.json").read_text())["label_counts"][split]
    if (set(counts) != set(LABELS) or any(type(n) is not int or n < 0 for n in counts.values())
            or sum(counts.values()) != len(proba)):
        raise ValueError(f"Invalid aggregate counts for {task}/{split}")
    return constrained(proba, counts)


def infer(data, artifact, split, prefix_only):
    def probability(task, x):
        return sum(w * aligned(m, x) for m, w in artifact["members"][task])

    x = features(data, PREFIX, split)
    p = probability(PREFIX, x)
    results = {PREFIX: dict(zip(x.index, count_assign(data, PREFIX, split, p)))}
    if not prefix_only:
        f = features(data, STATIC, split)
        if set(f.index) != set(x.index):
            raise ValueError("Full and prefix flow IDs do not match")
        p_aligned = pd.DataFrame(p, index=x.index).loc[f.index].to_numpy()
        fused = 0.70 * p_aligned + 0.30 * probability(STATIC, f)
        results[STATIC] = dict(zip(f.index, count_assign(data, STATIC, split, fused)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit", choices=("train", "all"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--prefix-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.fit:
            splits = ["train"] if args.fit == "train" else ["train", "validation"]
            if args.split in splits:
                parser.error("Cannot score validation on a model fitted using validation labels")
            tasks = [PREFIX] if args.prefix_only else [PREFIX, STATIC]
            artifact = {"kind": "public-data-reference", "fit_splits": splits,
                        "members": {t: fit(args.data, t, splits) for t in tasks}}
            joblib.dump(artifact, args.model, compress=3)
        else:
            artifact = joblib.load(args.model)  # Load only trusted checkpoints.
        if artifact["kind"] != "public-data-reference" or args.split in artifact["fit_splits"]:
            parser.error("Wrong checkpoint type or evaluation split was used in fitting")
        results = infer(args.data, artifact, args.split, args.prefix_only)
    args.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
    if args.split == "validation":
        for task, pred in results.items():
            truth = labels(args.data, task, args.split, pd.Index(pred))
            print(json.dumps({"task": task,
                "macro_f1": f1_score(truth, list(pred.values()), average="macro", labels=LABELS),
                "accuracy": accuracy_score(truth, list(pred.values()))}))


if __name__ == "__main__":
    main()
