#!/usr/bin/env python3
"""Search robust Task-1 fusion models for Challenge 495.

Task 2 is modeled strictly from prefix-five features. Task 1 may use both the
full-flow and prefix-five views of the same flow. The script reports validation
and repeated-CV performance, then emits probabilities for candidate v4 models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
OUT = ROOT / "v4_output"
LABELS = np.array(["facebook", "google", "instagram", "mozilla", "x", "youtube"])
SEEDS = (17, 42, 105)


def load_view(task: str, split: str, labeled: bool = True) -> pd.DataFrame:
    base = DATA / task
    features = pd.read_csv(base / f"{split}_features.csv")
    if not labeled:
        return features
    labels = pd.read_csv(base / f"{split}_labels.csv")
    return features.merge(labels, on="flow_id", how="inner", validate="one_to_one")


def merge_views(split: str, labeled: bool = True) -> pd.DataFrame:
    full = load_view("task1_static", split, labeled=labeled)
    prefix = load_view("task2_realtime_prefix5", split, labeled=labeled)
    if labeled:
        if not full[["flow_id", "app"]].sort_values("flow_id").reset_index(drop=True).equals(
            prefix[["flow_id", "app"]].sort_values("flow_id").reset_index(drop=True)
        ):
            raise ValueError(f"{split}: labels differ between task views")
        prefix = prefix.drop(columns="app")
    return full.merge(prefix, on="flow_id", suffixes=("_full", "_p5"), validate="one_to_one")


def engineer_single(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.drop(columns=["flow_id", "split", "app"], errors="ignore").copy()
    x["transport"] = (x["transport"].astype(str).str.lower() == "udp").astype(np.int8)
    for column in list(x.columns):
        if column not in {"transport", "start_rel", "server_port"}:
            values = pd.to_numeric(x[column], errors="coerce").fillna(0.0)
            x[f"{column}_log"] = np.log1p(np.maximum(values, 0.0))
    x["total_packets"] = x["packet_count_up"] + x["packet_count_down"]
    x["total_bytes"] = x["bytes_up"] + x["bytes_down"]
    x["down_packet_fraction"] = x["packet_count_down"] / (x["total_packets"] + 1.0)
    x["down_byte_fraction"] = x["bytes_down"] / (x["total_bytes"] + 1.0)
    x["bytes_per_packet"] = x["total_bytes"] / (x["total_packets"] + 1.0)
    return x.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def engineer_paired(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.drop(columns=["flow_id", "split_full", "split_p5", "app"], errors="ignore").copy()
    for col in ["transport_full", "transport_p5"]:
        x[col] = (x[col].astype(str).str.lower() == "udp").astype(np.int8)

    common = sorted(
        set(c.removesuffix("_full") for c in x if c.endswith("_full"))
        & set(c.removesuffix("_p5") for c in x if c.endswith("_p5"))
    )
    for base in common:
        a = pd.to_numeric(x[f"{base}_full"], errors="coerce").fillna(0.0)
        b = pd.to_numeric(x[f"{base}_p5"], errors="coerce").fillna(0.0)
        x[f"{base}_tail"] = a - b
        x[f"{base}_full_to_p5"] = a / (np.abs(b) + 1.0)

    for column in list(x.columns):
        if "transport" not in column and "server_port" not in column and "start_rel" not in column:
            values = pd.to_numeric(x[column], errors="coerce").fillna(0.0)
            x[f"{column}_signed_log"] = np.sign(values) * np.log1p(np.abs(values))
    return x.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def aligned(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(LABELS)), dtype=float)
    for source, label in enumerate(model.classes_):
        target = int(np.flatnonzero(LABELS == label)[0])
        out[:, target] = raw[:, source]
    return out


def model_set(view: str, seed: int):
    if view == "prefix":
        return [
            RandomForestClassifier(
                n_estimators=700, min_samples_leaf=1, max_features="sqrt",
                class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
            ),
            ExtraTreesClassifier(
                n_estimators=700, min_samples_leaf=1, max_features="sqrt",
                class_weight="balanced", random_state=seed, n_jobs=-1,
            ),
        ], [0.4, 0.6]

    if view == "full":
        return [
            LGBMClassifier(
                n_estimators=500, learning_rate=0.025, num_leaves=7,
                min_child_samples=8, class_weight="balanced", reg_lambda=2.0,
                reg_alpha=0.2, verbosity=-1, random_state=seed, n_jobs=-1,
            ),
            ExtraTreesClassifier(
                n_estimators=900, min_samples_leaf=2, max_features=0.9,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            ),
            CatBoostClassifier(
                iterations=500, depth=5, learning_rate=0.035, loss_function="MultiClass",
                auto_class_weights="Balanced", l2_leaf_reg=5.0, random_seed=seed,
                verbose=False, allow_writing_files=False, thread_count=-1,
            ),
        ], [0.35, 0.35, 0.30]

    return [
        LGBMClassifier(
            n_estimators=500, learning_rate=0.025, num_leaves=7,
            min_child_samples=8, class_weight="balanced", reg_lambda=2.5,
            reg_alpha=0.25, verbosity=-1, random_state=seed, n_jobs=-1,
        ),
        ExtraTreesClassifier(
            n_estimators=900, min_samples_leaf=2, max_features=0.75,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        CatBoostClassifier(
            iterations=550, depth=5, learning_rate=0.03, loss_function="MultiClass",
            auto_class_weights="Balanced", l2_leaf_reg=6.0, random_seed=seed,
            verbose=False, allow_writing_files=False, thread_count=-1,
        ),
    ], [0.35, 0.35, 0.30]


def fit_predict(view: str, train_x, train_y, target_x, seeds=SEEDS) -> np.ndarray:
    by_seed = []
    for seed in seeds:
        models, weights = model_set(view, seed)
        blended = np.zeros((len(target_x), len(LABELS)), dtype=float)
        for model, weight in zip(models, weights):
            model.fit(train_x, train_y)
            blended += weight * aligned(model, target_x)
        by_seed.append(blended)
    return np.mean(by_seed, axis=0)


def constrained(prob: np.ndarray, counts: dict[str, int]) -> np.ndarray:
    slots = np.concatenate(
        [np.repeat(i, int(counts[label])) for i, label in enumerate(LABELS)]
    )
    rows, columns = linear_sum_assignment(-np.log(np.clip(prob[:, slots], 1e-12, 1.0)))
    assigned = np.empty(len(prob), dtype=int)
    assigned[rows] = slots[columns]
    return LABELS[assigned]


def metrics(y, prob, counts=None):
    pred = LABELS[prob.argmax(axis=1)]
    result = {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
    }
    if counts is not None:
        cpred = constrained(prob, counts)
        result["constrained_macro_f1"] = float(f1_score(y, cpred, average="macro"))
        result["constrained_accuracy"] = float(accuracy_score(y, cpred))
    return result


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train_full = load_view("task1_static", "train")
    val_full = load_view("task1_static", "validation")
    train_prefix = load_view("task2_realtime_prefix5", "train")
    val_prefix = load_view("task2_realtime_prefix5", "validation")
    train_pair = merge_views("train")
    val_pair = merge_views("validation")
    y_train = train_full["app"].to_numpy(str)
    y_val = val_full["app"].to_numpy(str)
    val_counts = pd.Series(y_val).value_counts().to_dict()

    x_train = {
        "prefix": engineer_single(train_prefix),
        "full": engineer_single(train_full),
        "paired": engineer_paired(train_pair),
    }
    x_val = {
        "prefix": engineer_single(val_prefix),
        "full": engineer_single(val_full),
        "paired": engineer_paired(val_pair),
    }
    probs = {view: fit_predict(view, x_train[view], y_train, x_val[view]) for view in x_train}

    candidates = {}
    for wp in np.arange(0.0, 1.01, 0.1):
        for wf in np.arange(0.0, 1.01 - wp, 0.1):
            wpair = round(1.0 - wp - wf, 10)
            if wpair < -1e-9:
                continue
            name = f"p{wp:.1f}_f{wf:.1f}_x{wpair:.1f}"
            candidates[name] = wp * probs["prefix"] + wf * probs["full"] + wpair * probs["paired"]

    validation = {name: metrics(y_val, prob, val_counts) for name, prob in candidates.items()}
    ranking = sorted(validation, key=lambda n: validation[n]["macro_f1"], reverse=True)

    # Repeated CV on all released labels checks whether validation leaders are stable.
    all_full = pd.concat([train_full, val_full], ignore_index=True)
    all_prefix = pd.concat([train_prefix, val_prefix], ignore_index=True)
    all_pair = pd.concat([train_pair, val_pair], ignore_index=True)
    all_y = all_full["app"].to_numpy(str)
    all_x = {
        "prefix": engineer_single(all_prefix),
        "full": engineer_single(all_full),
        "paired": engineer_paired(all_pair),
    }
    shortlist = ranking[:8]
    cv_scores = {name: [] for name in shortlist}
    splitter = RepeatedStratifiedKFold(n_splits=4, n_repeats=2, random_state=2026)
    for fold, (tr, va) in enumerate(splitter.split(all_x["full"], all_y)):
        fold_probs = {}
        for view in all_x:
            fold_probs[view] = fit_predict(
                view, all_x[view].iloc[tr], all_y[tr], all_x[view].iloc[va], seeds=(42,)
            )
        for name in shortlist:
            parts = name.split("_")
            wp = float(parts[0][1:])
            wf = float(parts[1][1:])
            wx = float(parts[2][1:])
            mixed = wp * fold_probs["prefix"] + wf * fold_probs["full"] + wx * fold_probs["paired"]
            pred = LABELS[mixed.argmax(axis=1)]
            cv_scores[name].append(float(f1_score(all_y[va], pred, average="macro")))

    summary = []
    for name in shortlist:
        row = {"name": name, **validation[name]}
        row["cv_mean"] = float(np.mean(cv_scores[name]))
        row["cv_std"] = float(np.std(cv_scores[name]))
        row["robust_score"] = row["cv_mean"] - 0.25 * row["cv_std"]
        summary.append(row)
    summary.sort(key=lambda r: (r["robust_score"], r["macro_f1"]), reverse=True)

    # Fit all public labels and emit test probabilities for the best robust blend.
    best = summary[0]["name"]
    parts = best.split("_")
    weights = [float(parts[0][1:]), float(parts[1][1:]), float(parts[2][1:])]
    test_full = load_view("task1_static", "test", labeled=False)
    test_prefix = load_view("task2_realtime_prefix5", "test", labeled=False)
    test_pair = merge_views("test", labeled=False)
    test_x = {
        "prefix": engineer_single(test_prefix),
        "full": engineer_single(test_full),
        "paired": engineer_paired(test_pair),
    }
    test_probs = {
        view: fit_predict(view, all_x[view], all_y, test_x[view]) for view in all_x
    }
    fused = weights[0] * test_probs["prefix"] + weights[1] * test_probs["full"] + weights[2] * test_probs["paired"]

    np.savez_compressed(
        OUT / "probabilities.npz",
        flow_id=test_full["flow_id"].astype(str).to_numpy(),
        labels=LABELS,
        prefix=test_probs["prefix"],
        full=test_probs["full"],
        paired=test_probs["paired"],
        fused=fused,
    )
    report = {
        "best": best,
        "weights": {"prefix": weights[0], "full": weights[1], "paired": weights[2]},
        "validation_top": [{"name": n, **validation[n]} for n in ranking[:12]],
        "robust_shortlist": summary,
        "prefix_validation": metrics(y_val, probs["prefix"], val_counts),
        "full_validation": metrics(y_val, probs["full"], val_counts),
        "paired_validation": metrics(y_val, probs["paired"], val_counts),
    }
    (OUT / "experiment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
