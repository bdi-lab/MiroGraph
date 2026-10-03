import argparse
import json
import os
import subprocess
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classifier_designer import (
    _agent_value,
    _score_value,
    active_feature_counts,
    active_fields,
    extract_feature_matrices,
    load_scores,
    load_feature_schema,
    selected_agents,
    set_active_feature_schema,
    split_train_val,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


MODE_TO_EXCLUDED_AGENT = {
    "no_structure": "structure",
    "no_semantic": "semantic",
    "no_trend": "trend",
    "single_score": "",
}

SINGLE_SCORE_FEATURE_SCHEMA: Dict[str, Any] = {
    "version": 1,
    "task": "high-impact future scientific concept-pair prediction",
    "agents": {
        agent: {
            "features": [
                {
                    "name": "score",
                    "description": "Single scalar score emitted by the agent.",
                    "direction": "higher means stronger evidence for a future high-impact pair",
                    "value_range": "0 to 1",
                    "source": "llm",
                }
            ]
        }
        for agent in ("structure", "semantic", "trend")
    },
}


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def with_suffix(base: str, suffix: str) -> str:
    root, ext = os.path.splitext(base)
    if ext:
        return f"{root}_{suffix}{ext}"
    return f"{base}_{suffix}"


def mode_path(base: str | None, mode: str, single_mode: bool) -> str | None:
    if not base:
        return None
    return base if single_mode else with_suffix(base, mode)


def load_prediction_metrics(path: str) -> Dict[str, float]:
    labels: List[int] = []
    preds: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "ground_truth_label" not in row:
                continue
            labels.append(int(row["ground_truth_label"]))
            preds.append(int(row["meta_code_pred"]))
    if not labels:
        return {}
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=[0, 1], zero_division=0)),
        "samples": float(len(labels)),
    }


def schema_for_mode(args: argparse.Namespace, mode: str) -> Dict[str, Any]:
    if mode == "single_score":
        return SINGLE_SCORE_FEATURE_SCHEMA
    if not args.feature_schema:
        raise ValueError("--feature_schema is required except for --mode single_score.")
    return load_feature_schema(args.feature_schema)


def schema_path_for_mode(args: argparse.Namespace, mode: str, output: str) -> str:
    if mode != "single_score":
        if not args.feature_schema:
            raise ValueError("--feature_schema is required except for --mode single_score.")
        return args.feature_schema
    parent = os.path.dirname(output) or "."
    os.makedirs(parent, exist_ok=True)
    path = os.path.join(parent, "single_score_feature_schema.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(SINGLE_SCORE_FEATURE_SCHEMA, f, ensure_ascii=False, indent=2)
    return path


def parse_int_list(value: str) -> List[int]:
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def parse_float_list(value: str) -> List[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Expected at least one float value.")
    return out


def parse_modes(args: argparse.Namespace) -> List[str]:
    if args.classifier_backend == "mlp":
        if args.mode or args.modes:
            raise ValueError("The MLP classifier ablation does not use --mode or --modes.")
        return ["mlp"]
    if args.modes:
        modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    elif args.mode:
        modes = [args.mode]
    else:
        raise ValueError("Either --mode or --modes must be provided.")
    unknown = [mode for mode in modes if mode not in MODE_TO_EXCLUDED_AGENT]
    if unknown:
        raise ValueError(f"Unknown ablation mode(s): {unknown}")
    return modes


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    thresholds = np.unique(np.r_[np.linspace(0.05, 0.95, 181), scores])
    best_threshold = 0.5
    best_metrics = {"accuracy": -1.0, "macro_f1": -1.0}
    best_key = (-1.0, -1.0)
    for threshold in thresholds:
        preds = (scores >= float(threshold)).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "macro_f1": float(f1_score(y_true, preds, average="macro", labels=[0, 1], zero_division=0)),
        }
        key = (metrics["macro_f1"], metrics["accuracy"])
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def make_mlp_model(
    hidden_units: int,
    alpha: float,
    learning_rate_init: float,
    activation: str,
    seed: int,
    max_iter: int,
):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=seed,
        ),
    )


def fit_select_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Any, float, Dict[str, Any]]:
    hidden_values = parse_int_list(args.mlp_hidden_units)
    alpha_values = parse_float_list(args.mlp_alpha_values)
    lr_values = parse_float_list(args.mlp_learning_rates)
    activations = [x.strip() for x in args.mlp_activations.split(",") if x.strip()]
    if not activations:
        raise ValueError("--mlp_activations must contain at least one activation.")

    best: Tuple[Tuple[float, float], Any, float, Dict[str, Any]] | None = None
    candidate_records = []
    candidate_id = 0
    for hidden_units in hidden_values:
        for alpha in alpha_values:
            for lr in lr_values:
                for activation in activations:
                    candidate_id += 1
                    model = make_mlp_model(
                        hidden_units=hidden_units,
                        alpha=alpha,
                        learning_rate_init=lr,
                        activation=activation,
                        seed=args.seed + candidate_id,
                        max_iter=args.mlp_max_iter,
                    )
                    try:
                        model.fit(X_train, y_train)
                        scores = model.predict_proba(X_val)[:, 1]
                        threshold, metrics = tune_threshold(y_val, scores)
                    except Exception as exc:
                        candidate_records.append(
                            {
                                "candidate_id": candidate_id,
                                "hidden_units": hidden_units,
                                "alpha": alpha,
                                "learning_rate_init": lr,
                                "activation": activation,
                                "error": str(exc),
                            }
                        )
                        continue
                    record = {
                        "candidate_id": candidate_id,
                        "hidden_units": hidden_units,
                        "alpha": alpha,
                        "learning_rate_init": lr,
                        "activation": activation,
                        "threshold": float(threshold),
                        "val_accuracy": float(metrics["accuracy"]),
                        "val_macro_f1": float(metrics["macro_f1"]),
                    }
                    candidate_records.append(record)
                    key = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
                    if best is None or key > best[0]:
                        best = (key, model, float(threshold), record)

    if best is None:
        raise RuntimeError("No valid MLP candidate could be trained.")

    _, _, threshold, selected = best
    final_model = make_mlp_model(
        hidden_units=int(selected["hidden_units"]),
        alpha=float(selected["alpha"]),
        learning_rate_init=float(selected["learning_rate_init"]),
        activation=str(selected["activation"]),
        seed=args.seed,
        max_iter=args.mlp_max_iter,
    )
    final_model.fit(np.vstack([X_train, X_val]), np.r_[y_train, y_val])
    info = {
        "kind": "mlp",
        "architecture": "single_hidden_layer",
        "selected": selected,
        "threshold": float(threshold),
        "num_candidates": len(candidate_records),
        "candidates": candidate_records,
    }
    return final_model, threshold, info


def save_predictions(
    path: str,
    test_records: List[Dict[str, Any]],
    preds: np.ndarray,
    selected_feature_agents: Tuple[str, ...],
) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for record, yhat in zip(test_records, preds):
            out = {
                "concept1": record.get("concept1"),
                "concept2": record.get("concept2"),
                "meta_code_pred": int(yhat),
                "meta_code_pred_label": "Yes" if int(yhat) == 1 else "No",
            }
            for agent in selected_feature_agents:
                agent_record = record.get(agent, {})
                if not isinstance(agent_record, dict):
                    agent_record = {}
                for field in active_fields(agent):
                    if agent == "structure":
                        out[f"{agent}_{field}"] = _agent_value(agent_record.get(field, 0.0))
                    else:
                        out[f"{agent}_{field}"] = _score_value(agent_record.get(field, 0.0))
            if "ground_truth_label" in record and record["ground_truth_label"] is not None:
                out["ground_truth_label"] = int(record["ground_truth_label"])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def run_mlp(args: argparse.Namespace, mode: str, single_mode: bool) -> Dict[str, Any]:
    output = mode_path(args.output, mode, single_mode)
    assert output is not None

    excluded_agent = ""
    exclude_agents: Tuple[str, ...] = ()
    schema = schema_for_mode(args, mode)
    set_active_feature_schema(schema, exclude_agents)

    train_records = load_scores(args.train_scores_cache)
    test_records = load_scores(args.test_scores_cache)
    X_train, y_train, X_test, y_test, names = extract_feature_matrices(train_records, test_records, "agent_features")
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, args.train_ratio, args.val_ratio, args.seed)

    print("[ABLATION] mode=mlp backend=mlp")
    print(
        f"[INFO] Score cache: {len(X_train)}, Train split: {len(X_tr)}, "
        f"Val split: {len(X_val)}, Test samples: {len(X_test)}, Features: {len(names)}"
    )
    counts = active_feature_counts()
    print(f"[FEATURES] {', '.join(f'{agent}={counts[agent]}' for agent in selected_agents())}, total={len(names)}")

    model, threshold, info = fit_select_mlp(X_tr, y_tr, X_val, y_val, args)
    scores = model.predict_proba(X_test)[:, 1]
    preds = (scores >= threshold).astype(int)
    selected_feature_agents = selected_agents()
    if y_test is not None:
        info["test_metrics"] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "macro_f1": float(f1_score(y_test, preds, average="macro", labels=[0, 1], zero_division=0)),
        }
        print("[METRICS]")
        print(f"Accuracy: {info['test_metrics']['accuracy']:.4f}")
        print(f"Macro F1: {info['test_metrics']['macro_f1']:.4f}")
    else:
        print("[METRICS] skipped because test/eval labels are not available.")

    save_predictions(output, test_records, preds, selected_feature_agents)
    print(f"[DONE] saved to {output}")

    metrics = load_prediction_metrics(output)
    return {
        "mode": mode,
        "excluded_agent": excluded_agent,
        "backend": "mlp",
        "output": output,
        "codes_out": None,
        "best_code_out": None,
        "metrics": metrics,
    }


def run_classifier_designer(args: argparse.Namespace, mode: str, single_mode: bool) -> Dict[str, Any]:
    output = mode_path(args.output, mode, single_mode)
    codes_out = mode_path(args.codes_out, mode, single_mode)
    best_code_out = mode_path(args.best_code_out, mode, single_mode)
    assert output is not None

    feature_schema = schema_path_for_mode(args, mode, output)
    command = [
        sys.executable,
        "-u",
        "classifier_designer.py",
        "--train_scores_cache",
        args.train_scores_cache,
        "--test_scores_cache",
        args.test_scores_cache,
        "--feature_schema",
        feature_schema,
        "--output",
        output,
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--num_candidates",
        str(args.num_candidates),
        "--code_model",
        args.code_model,
        "--code_temperature",
        str(args.code_temperature),
        "--request_timeout",
        str(args.request_timeout),
        "--api_max_retries",
        str(args.api_max_retries),
        "--seed",
        str(args.seed),
    ]
    excluded_agent = MODE_TO_EXCLUDED_AGENT[mode]
    if excluded_agent:
        command.extend(["--exclude_agents", excluded_agent])
    if args.api_key:
        command.extend(["--api_key", args.api_key])
    if codes_out:
        command.extend(["--codes_out", codes_out])
    if best_code_out:
        command.extend(["--best_code_out", best_code_out])

    print(f"[ABLATION] mode={mode} excluded={excluded_agent or 'none'}")
    print("[COMMAND] " + " ".join(command))
    subprocess.run(command, check=True)

    metrics = load_prediction_metrics(output)
    result: Dict[str, Any] = {
        "mode": mode,
        "excluded_agent": excluded_agent,
        "backend": "classifier_designer",
        "output": output,
        "codes_out": codes_out,
        "best_code_out": best_code_out,
        "metrics": metrics,
    }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache-only feature ablations.")
    p.add_argument("--mode", choices=sorted(MODE_TO_EXCLUDED_AGENT), help="Single ablation mode.")
    p.add_argument("--modes", type=str, help="Comma-separated modes, e.g. no_structure,no_semantic,no_trend,single_score.")
    p.add_argument("--train_scores_cache", required=True, help="Existing train scores JSONL.")
    p.add_argument("--test_scores_cache", required=True, help="Existing test/eval scores JSONL.")
    p.add_argument("--feature_schema", default=None, help="Feature schema used to create the score cache. Not needed for --mode single_score.")
    p.add_argument("--api_key", default=None, help="OpenAI API key, otherwise read from environment.")
    p.add_argument("--output", required=True, help="Prediction JSONL path; suffixed by mode when multiple modes are run.")
    p.add_argument("--codes_out", default=None, help="Candidate code JSONL path; suffixed by mode when multiple modes are run.")
    p.add_argument("--best_code_out", default=None, help="Best code path; suffixed by mode when multiple modes are run.")
    p.add_argument("--train_ratio", type=float, default=70.0)
    p.add_argument("--val_ratio", type=float, default=15.0)
    p.add_argument("--num_candidates", type=int, default=10)
    p.add_argument("--code_model", default="gpt-5.4")
    p.add_argument("--code_temperature", type=float, default=0.6)
    p.add_argument("--request_timeout", type=float, default=180.0)
    p.add_argument("--api_max_retries", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary_out", default=None, help="Optional JSON summary for all modes.")
    p.add_argument(
        "--classifier_backend",
        choices=["classifier_designer", "mlp"],
        default="classifier_designer",
        help="Use LLM-generated classifier code or a fixed single-hidden-layer MLP classifier.",
    )
    p.add_argument("--mlp_hidden_units", default="8,16,32,64", help="Comma-separated hidden sizes for one hidden layer.")
    p.add_argument("--mlp_alpha_values", default="0.0001,0.001,0.01", help="Comma-separated L2 alpha values.")
    p.add_argument("--mlp_learning_rates", default="0.001,0.003", help="Comma-separated learning rates.")
    p.add_argument("--mlp_activations", default="relu,tanh", help="Comma-separated MLP activations.")
    p.add_argument("--mlp_max_iter", type=int, default=1200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    modes = parse_modes(args)
    single_mode = len(modes) == 1

    results = []
    for mode in modes:
        if args.classifier_backend == "mlp":
            results.append(run_mlp(args, mode, single_mode))
        else:
            results.append(run_classifier_designer(args, mode, single_mode))

    if args.summary_out:
        ensure_parent_dir(args.summary_out)
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[SUMMARY] saved to {args.summary_out}")

    print("[ABLATION_SUMMARY]")
    for result in results:
        metrics = result["metrics"]
        if metrics:
            print(
                f"{result['mode']}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"samples={int(metrics['samples'])}"
            )
        else:
            print(f"{result['mode']}: metrics unavailable")


if __name__ == "__main__":
    main()
