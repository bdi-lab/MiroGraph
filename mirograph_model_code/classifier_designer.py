import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from feature_schema import agent_features_from_schema, load_feature_schema
from mirograph_config import AGENTS, DEFAULT_CODE_MODEL, DEFAULT_CODE_TEMPERATURE

load_dotenv(override=True)

ACTIVE_FEATURE_SCHEMA: Dict[str, Any] = {}
ACTIVE_FEATURE_NAMES_BY_AGENT: Dict[str, Tuple[str, ...]] = {}
ACTIVE_SELECTED_AGENTS: Tuple[str, ...] = AGENTS


def set_active_feature_schema(
    schema: Dict[str, Any],
    exclude_agents: Tuple[str, ...] = (),
) -> None:
    global ACTIVE_FEATURE_SCHEMA, ACTIVE_FEATURE_NAMES_BY_AGENT, ACTIVE_SELECTED_AGENTS
    ACTIVE_FEATURE_SCHEMA = schema
    raw_fields = agent_features_from_schema(schema)
    excluded = set(exclude_agents)
    unknown = sorted(excluded - set(AGENTS))
    if unknown:
        raise ValueError(f"Unknown excluded agent(s): {unknown}")
    ACTIVE_FEATURE_NAMES_BY_AGENT = {
        agent: tuple(fields)
        for agent, fields in raw_fields.items()
    }
    ACTIVE_SELECTED_AGENTS = tuple(agent for agent in AGENTS if agent not in excluded)
    if not ACTIVE_SELECTED_AGENTS:
        raise ValueError("At least one agent must remain after --exclude_agents.")
    empty_agents = [agent for agent in ACTIVE_SELECTED_AGENTS if not ACTIVE_FEATURE_NAMES_BY_AGENT[agent]]
    if empty_agents:
        raise ValueError(f"No active feature fields remain for selected agent(s): {empty_agents}")


def active_fields(agent: str) -> Tuple[str, ...]:
    if agent not in ACTIVE_FEATURE_NAMES_BY_AGENT:
        raise RuntimeError("Active feature schema is not loaded. Pass --feature_schema.")
    return ACTIVE_FEATURE_NAMES_BY_AGENT[agent]


def active_feature_counts() -> Dict[str, int]:
    return {agent: len(active_fields(agent)) for agent in ACTIVE_SELECTED_AGENTS}


def selected_agents() -> Tuple[str, ...]:
    return ACTIVE_SELECTED_AGENTS


def load_scores(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _agent_value(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        out = 0.0
    if not np.isfinite(out):
        out = 0.0
    return float(out)


def _score_value(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        out = 0.0
    if not np.isfinite(out):
        out = 0.0
    return float(max(0.0, min(1.0, out)))


def agent_feature_vector(record: Dict[str, Any]) -> List[float]:
    values: List[float] = []
    for agent in selected_agents():
        agent_record = record.get(agent, {})
        if not isinstance(agent_record, dict):
            agent_record = {}
        for field in active_fields(agent):
            if agent == "structure":
                values.append(_agent_value(agent_record.get(field, 0.0)))
            else:
                values.append(_score_value(agent_record.get(field, 0.0)))
    return values


def agent_feature_names() -> List[str]:
    return [
        f"{agent}_{field}"
        for agent in selected_agents()
        for field in active_fields(agent)
    ]


DEFAULT_FEATURE_SET = "agent_features"


def build_feature_vector(record: Dict[str, Any], feature_set: str = DEFAULT_FEATURE_SET) -> List[float]:
    if feature_set == "agent_features":
        return agent_feature_vector(record)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def feature_names(feature_set: str = DEFAULT_FEATURE_SET) -> List[str]:
    if feature_set == "agent_features":
        return agent_feature_names()
    raise ValueError(f"Unknown feature_set: {feature_set}")


def extract_xy(
    records: List[Dict[str, Any]],
    require_labels: bool = False,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X, y = [], []
    labels_present = True
    for r in records:
        X.append(build_feature_vector(r, feature_set))
        if "ground_truth_label" in r and r["ground_truth_label"] is not None:
            y.append(int(r["ground_truth_label"]))
        else:
            labels_present = False
    if require_labels and not labels_present:
        raise ValueError("All training score records must include ground_truth_label.")
    y_arr = np.array(y, dtype=int) if labels_present else None
    return np.array(X, dtype=float), y_arr


def extract_feature_matrices(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    feature_set: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    X_train, y_train = extract_xy(train_records, require_labels=True, feature_set=feature_set)
    X_test, y_test = extract_xy(test_records, require_labels=False, feature_set=feature_set)
    return X_train, y_train, X_test, y_test, feature_names(feature_set)


def format_column_spec(names: List[str]) -> str:
    if names == agent_feature_names():
        lines = []
        start = 0
        for agent in selected_agents():
            fields = active_fields(agent)
            end = start + len(fields) - 1
            lines.append(f"{start}..{end} {agent}: {', '.join(fields)}")
            start = end + 1
        return "\n".join(lines)
    if len(names) <= 80:
        return "\n".join(f"- {i}: {name}" for i, name in enumerate(names))
    head = "\n".join(f"- {i}: {name}" for i, name in enumerate(names[:20]))
    return (
        f"{head}\n"
        f"- 20..{len(names) - 1}: additional numeric features extracted from the score cache"
    )


def feature_guidance(names: List[str]) -> str:
    agent_names = agent_feature_names()
    if names == agent_names:
        agents_text = ", ".join(selected_agents())
        details = [
            f"Use the {len(names)} selected agent feature columns as the complete model input.",
            f"Selected agents: {agents_text}.",
            f"Any preprocessing must preserve exactly these {len(names)} columns.",
        ]
        if "structure" in selected_agents():
            details.append("The structure columns are graph-derived SCG measurements and should be treated as the Structural Agent output.")
        else:
            details.append("This ablation intentionally excludes structural columns; do not assume graph-derived structure features are present.")
        return " ".join(details)
    return """Use the provided numeric columns as the model input."""


def meta_code_prompt(num_candidates: int, names: List[str]) -> str:
    column_spec = format_column_spec(names)
    guidance = feature_guidance(names)
    if num_candidates <= 2:
        anchor_instruction = """
For this small candidate set, provide concise implementations that still perform internal validation for
threshold or hyperparameter choices when appropriate."""
    else:
        anchor_instruction = """
Generate diverse implementations by varying preprocessing, regularization, model selection, and
threshold-selection choices."""
    return f"""You are an expert Python ML engineer.
Generate {num_candidates} diverse Python implementations of fit_and_predict(train_scores, train_labels, test_scores).
Each implementation must define fit_and_predict(train_scores, train_labels, test_scores) returning a list of 0/1 predictions for each row in test_scores.
Input shapes:
- train_scores: numpy array shape (N,{len(names)}) with numeric features
- train_labels: numpy array shape (N,) with 0/1
- test_scores: numpy array shape (M,{len(names)}) with numeric features
Column order:
{column_spec}
{guidance}
Task details:
- The labels are balanced high-impact scientific concept-pair labels.
- Columns are noisy but meaningful evidence from the MiroGraph scoring stage.
- Use only the provided numeric columns. Do not create external graph, text, lexical, or dataset-derived features.
{anchor_instruction}
Optimization target:
- Prioritize macro-F1 and accuracy, not AUC.
Rules:
- You may use any Python library that is commonly available in data science stacks (e.g., numpy, sklearn, scipy). Prefer strong predictive performance over minimal dependencies.
- Each implementation should choose hyperparameters and thresholds using only train_scores and train_labels.
- Scaling, imputation, calibration, and model selection are allowed.
- Use only train_scores and train_labels for fitting or threshold selection. Do not assume test labels exist.
- Return JSON ONLY:
{{
  "codes": [
    {{"id": 1, "code": "<python code for fit_and_predict>"}}
    ...
  ]
}}
Do NOT wrap code in markdown fences. Provide exactly {num_candidates} entries in the codes list."""


def clean_code_text(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("python", "", 1).strip()
    return t


def single_code_prompt(names: List[str]) -> str:
    column_spec = format_column_spec(names)
    guidance = feature_guidance(names)
    return f"""You are an expert Python ML engineer.
Write a single Python implementation of fit_and_predict(train_scores, train_labels, test_scores).
Input shapes:
- train_scores: numpy array shape (N,{len(names)}) with numeric features
- train_labels: numpy array shape (N,) with 0/1
- test_scores: numpy array shape (M,{len(names)}) with numeric features
Column order:
{column_spec}
{guidance}
The labels are balanced high-impact scientific concept-pair labels. Prefer macro-F1 and accuracy.
Use only the provided numeric columns. Do not create external graph, text, lexical, or dataset-derived features.
Rules:
- You may use any Python library that is commonly available in data science stacks (e.g., numpy, sklearn, scipy). Prefer strong predictive performance over minimal dependencies.
- Do internal validation using only train_scores and train_labels when choosing thresholds, fusion weights, or hyperparameters.
- Scaling, imputation, calibration, and model selection are allowed.
- Use only train_scores and train_labels for fitting or threshold selection. Do not assume test labels exist.
Return ONLY executable Python code (no JSON, no markdown)."""


def generate_single_code(api_key: str, names: List[str], model: str, temperature: float, request_timeout: float, api_max_retries: int) -> str:
    client = OpenAI(api_key=api_key, timeout=request_timeout, max_retries=api_max_retries)
    system = "You write runnable Python code. Output only code."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": single_code_prompt(names)},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    return clean_code_text(content)


def generate_model_codes(api_key: str, num_candidates: int, names: List[str], model: str, temperature: float, request_timeout: float, api_max_retries: int) -> List[str]:
    client = OpenAI(api_key=api_key, timeout=request_timeout, max_retries=api_max_retries)
    system = "You write runnable Python code. Output only code."
    user = meta_code_prompt(num_candidates, names)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    codes: List[str] = []
    try:
        payload = json.loads(content)
        codes_raw = payload.get("codes", [])
        for entry in codes_raw:
            if not isinstance(entry, dict):
                continue
            code_text = entry.get("code") or ""
            code_text = clean_code_text(code_text)
            if code_text:
                codes.append(code_text)
    except Exception:
        codes_raw = []
    if not codes:
        parts = content.split("\n\n")
        for p in parts:
            p = clean_code_text(p)
            if "def fit_and_predict" in p:
                codes.append(p)
    return codes


def run_generated_code(code: str, X_train, y_train, X_test):
    ns: Dict[str, Any] = {}
    exec(code, ns, ns)
    if "fit_and_predict" not in ns:
        raise RuntimeError("Generated code missing fit_and_predict")
    preds = ns["fit_and_predict"](X_train, y_train, X_test)
    return preds


def split_train_val(X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float, seed: int):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    total_ratio = train_ratio + val_ratio
    if total_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be positive.")
    n_val = int(round(n * (val_ratio / total_ratio)))
    n_val = min(max(n_val, 1), n - 1)
    n_train = n - n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    return X_train, y_train, X_val, y_val


def compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", labels=[0, 1], zero_division=0)),
    }


def validation_key(y_true: np.ndarray, preds: np.ndarray) -> Tuple[float, float]:
    metrics = compute_metrics(y_true, preds)
    return metrics["macro_f1"], metrics["accuracy"]


def coerce_binary_predictions(preds: Any, expected_len: int) -> np.ndarray:
    arr = np.asarray(preds).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(f"Expected {expected_len} predictions, got {len(arr)}")
    if arr.dtype.kind == "f":
        return (arr >= 0.5).astype(int)
    return (arr.astype(int) > 0).astype(int)


def candidate_sort_key(record: Dict[str, Any]) -> Tuple[float, float]:
    return float(record["val_macro_f1"]), float(record["val_accuracy"])


def parse_args():
    p = argparse.ArgumentParser(description="Meta LLM code generation for score-based prediction.")
    p.add_argument("--train_scores_cache", required=True, help="Path to train scores JSONL.")
    p.add_argument("--test_scores_cache", required=True, help="Path to test scores JSONL.")
    p.add_argument("--api_key", default=None, help="OpenAI API key.")
    p.add_argument("--output", default="results/metallm/meta_code_test_predictions.jsonl", help="Output JSONL.")
    p.add_argument("--codes_out", default=None, help="Optional JSONL file to save all generated code candidates with val accuracy.")
    p.add_argument("--best_code_out", default=None, help="Optional path to save the best code (plain text).")
    p.add_argument("--feature_schema", required=True, help="JSON schema generated by feature_designer.py.")
    p.add_argument("--exclude_agents", default="", help="Comma-separated agents to exclude, e.g. structure or semantic,trend.")
    p.add_argument("--train_ratio", type=float, default=70.0, help="Train split weight from train cache.")
    p.add_argument("--val_ratio", type=float, default=15.0, help="Validation split weight from train cache.")
    p.add_argument("--num_candidates", type=int, default=10, help="Number of code candidates to generate.")
    p.add_argument("--code_model", default=DEFAULT_CODE_MODEL, help="OpenAI model for classifier code generation.")
    p.add_argument("--code_temperature", type=float, default=DEFAULT_CODE_TEMPERATURE, help="Code generation temperature.")
    p.add_argument("--request_timeout", type=float, default=120.0, help="OpenAI request timeout in seconds.")
    p.add_argument("--api_max_retries", type=int, default=2, help="OpenAI client retry count.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    schema = load_feature_schema(args.feature_schema)
    exclude_agents = tuple(agent.strip() for agent in args.exclude_agents.split(",") if agent.strip())
    set_active_feature_schema(schema, exclude_agents)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    if args.num_candidates <= 0:
        raise ValueError("--num_candidates must be positive because classifier_designer generates classifier code via the LLM.")

    train_records = load_scores(args.train_scores_cache)
    test_records = load_scores(args.test_scores_cache)
    X_train, y_train, X_test, y_test, names = extract_feature_matrices(
        train_records,
        test_records,
        DEFAULT_FEATURE_SET,
    )
    assert y_train is not None

    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, args.train_ratio, args.val_ratio, args.seed)

    print(
        f"[INFO] Score cache: {len(X_train)}, Train split: {len(X_tr)}, "
        f"Val split: {len(X_val)}, Test samples: {len(X_test)}, Features: {len(names)}"
    )
    if args.feature_schema:
        print(f"[FEATURE_SCHEMA] {args.feature_schema}")
    if exclude_agents:
        print(f"[EXCLUDED_AGENTS] {', '.join(exclude_agents)}")
    counts = active_feature_counts()
    count_text = ", ".join(f"{agent}={counts[agent]}" for agent in selected_agents())
    print(f"[FEATURES] {count_text}, total={len(names)}")

    best_code = None
    code_records = []

    def add_candidate_record(idx: int, code: str, val_preds: np.ndarray, macro_f1_val: float, acc_val: float) -> None:
        position = len(code_records)
        rec = {
            "candidate_position": position,
            "index": idx,
            "val_accuracy": float(acc_val),
            "val_macro_f1": float(macro_f1_val),
            "code": code,
        }
        code_records.append(rec)

    best_key = (-1.0, -1.0)
    print(f"[INFO] Generating {args.num_candidates} code candidates with a single {args.code_model} call ...")
    codes = generate_model_codes(
        api_key,
        args.num_candidates,
        names,
        args.code_model,
        args.code_temperature,
        args.request_timeout,
        args.api_max_retries,
    )
    if not codes:
        print("[WARN] Bulk generation returned no parsable codes; falling back to per-code generation.")
        for idx in tqdm(range(args.num_candidates), desc="Code candidates (fallback)"):
            code = generate_single_code(
                api_key,
                names,
                args.code_model,
                args.code_temperature,
                args.request_timeout,
                args.api_max_retries,
            )
            if not code.strip():
                continue
            try:
                val_preds = run_generated_code(code, X_tr, y_tr, X_val)
                val_preds = coerce_binary_predictions(val_preds, len(X_val))
                macro_f1_val, acc_val = validation_key(y_val, val_preds)
            except Exception:
                continue
            add_candidate_record(idx, code, val_preds, macro_f1_val, acc_val)
            if (macro_f1_val, acc_val) > best_key:
                best_key = (macro_f1_val, acc_val)
                best_code = code

    for idx, code in enumerate(codes):
        try:
            val_preds = run_generated_code(code, X_tr, y_tr, X_val)
            val_preds = coerce_binary_predictions(val_preds, len(X_val))
            macro_f1_val, acc_val = validation_key(y_val, val_preds)
        except Exception:
            continue
        add_candidate_record(idx, code, val_preds, macro_f1_val, acc_val)
        if (macro_f1_val, acc_val) > best_key:
            best_key = (macro_f1_val, acc_val)
            best_code = code

    if best_code is None:
        for idx in range(args.num_candidates):
            code = generate_single_code(
                api_key,
                names,
                args.code_model,
                args.code_temperature,
                args.request_timeout,
                args.api_max_retries,
            )
            if not code.strip():
                continue
            try:
                val_preds = run_generated_code(code, X_tr, y_tr, X_val)
                val_preds = coerce_binary_predictions(val_preds, len(X_val))
                macro_f1_val, acc_val = validation_key(y_val, val_preds)
            except Exception:
                continue
            add_candidate_record(idx + len(code_records), code, val_preds, macro_f1_val, acc_val)
            if (macro_f1_val, acc_val) > best_key:
                best_key = (macro_f1_val, acc_val)
                best_code = code
        if best_code is None:
            raise RuntimeError("No valid generated code candidates produced a score.")

    ranked = sorted(code_records, key=candidate_sort_key, reverse=True)
    selected = ranked[0]
    best_code = selected["code"]
    selected_positions = {int(selected["candidate_position"])}
    for rec in code_records:
        rec["selected_for_final"] = int(rec["candidate_position"] in selected_positions)

    print("[SELECTED]")
    print(
        f"Candidate: {selected['candidate_position']}, "
        f"Val accuracy: {selected['val_accuracy']:.4f}, "
        f"Val macro-F1: {selected['val_macro_f1']:.4f}"
    )

    if args.codes_out:
        ensure_parent_dir(args.codes_out)
        with open(args.codes_out, "w", encoding="utf-8") as f:
            for rec in code_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[CODES] saved all candidates to {args.codes_out}")

    if args.best_code_out and best_code:
        ensure_parent_dir(args.best_code_out)
        with open(args.best_code_out, "w", encoding="utf-8") as f:
            f.write(best_code)
        print(f"[BEST] saved best code to {args.best_code_out}")

    preds = coerce_binary_predictions(run_generated_code(best_code, X_train, y_train, X_test), len(X_test))

    metrics = compute_metrics(y_test, preds) if y_test is not None else None
    if metrics:
        print("[METRICS]")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
    else:
        print("[METRICS] skipped because test/eval labels are not available.")

    ensure_parent_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        for r, yhat in zip(test_records, preds):
            out = {
                "concept1": r.get("concept1"),
                "concept2": r.get("concept2"),
                "meta_code_pred": int(yhat),
                "meta_code_pred_label": "Yes" if int(yhat) == 1 else "No",
            }
            for agent in selected_agents():
                agent_record = r.get(agent, {})
                if not isinstance(agent_record, dict):
                    agent_record = {}
                for field in active_fields(agent):
                    if agent == "structure":
                        out[f"{agent}_{field}"] = _agent_value(agent_record.get(field, 0.0))
                    else:
                        out[f"{agent}_{field}"] = _score_value(agent_record.get(field, 0.0))
            if "ground_truth_label" in r and r["ground_truth_label"] is not None:
                out["ground_truth_label"] = int(r["ground_truth_label"])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[DONE] saved to {args.output}")


if __name__ == "__main__":
    main()
