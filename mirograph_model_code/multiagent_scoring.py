import argparse
import importlib.util
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import extract_path
from feature_schema import agent_features_from_schema, feature_descriptions, load_feature_schema
from mirograph_config import AGENTS, DEFAULT_SCORING_MODEL, DEFAULT_SCORING_TEMPERATURE, SUPPORTED_DATASETS

load_dotenv(override=True)

ACTIVE_FEATURE_SCHEMA: Dict[str, Any] = {}
ACTIVE_FEATURE_NAMES_BY_AGENT: Dict[str, Tuple[str, ...]] = {}


def set_active_feature_schema(schema: Dict[str, Any]) -> None:
    global ACTIVE_FEATURE_SCHEMA, ACTIVE_FEATURE_NAMES_BY_AGENT
    ACTIVE_FEATURE_SCHEMA = schema
    ACTIVE_FEATURE_NAMES_BY_AGENT = agent_features_from_schema(schema)


def active_fields(agent: str) -> Tuple[str, ...]:
    if agent not in ACTIVE_FEATURE_NAMES_BY_AGENT:
        raise RuntimeError("Active feature schema is not loaded. Pass --feature_schema.")
    return ACTIVE_FEATURE_NAMES_BY_AGENT[agent]


def load_structure_feature_function(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Structural feature code not found: {path}")
    spec = importlib.util.spec_from_file_location("generated_structural_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import structural feature code: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "compute_structure_features", None)
    if not callable(fn):
        raise RuntimeError(f"{path} must define compute_structure_features(G, u, v, snapshot_year).")
    names = getattr(module, "FEATURE_NAMES", None)
    if list(names or []) != list(active_fields("structure")):
        raise RuntimeError("Structural code FEATURE_NAMES does not match the active feature schema.")
    return fn


def load_concept_graph_csv(csv_path: str) -> Tuple[nx.Graph, int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"concept_graph.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"concept1", "concept2", "year", "citation_count"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} must have columns {sorted(required)}")
    df["year"] = df["year"].astype(int)
    df["citation_count"] = df["citation_count"].astype(int)
    snapshot_year = int(df["year"].max())
    agg: Dict[Tuple[str, str, int], int] = {}
    for row in df.itertuples(index=False):
        c1, c2, y, cc = row.concept1, row.concept2, int(row.year), int(row.citation_count)
        a, b = (c1, c2) if str(c1) <= str(c2) else (c2, c1)
        k = (a, b, y)
        agg[k] = agg.get(k, 0) + cc
    edge_map: Dict[Tuple[str, str], Dict[int, int]] = {}
    for (a, b, y), cc in agg.items():
        edge_map.setdefault((a, b), {})
        edge_map[(a, b)][y] = edge_map[(a, b)].get(y, 0) + cc
    G = nx.Graph()
    for (a, b), year_dict in edge_map.items():
        G.add_node(a)
        G.add_node(b)
        G.add_edge(a, b, citations_by_year=dict(year_dict))
    return G, snapshot_year


def edge_year_dict(G: nx.Graph, u: str, v: str) -> Dict[int, int]:
    if not G.has_edge(u, v):
        return {}
    data = G.get_edge_data(u, v) or {}
    yd = data.get("citations_by_year", {})
    return dict(yd) if isinstance(yd, dict) else {}


def compute_structure_features(G: nx.Graph, u: str, v: str, snapshot_year: int, structure_fn=None) -> Dict[str, float]:
    if structure_fn is None:
        raise RuntimeError("Structural feature code is required. Pass --structure_code or set structure_code_path in --feature_schema.")
    raw = structure_fn(G, u, v, snapshot_year)
    if not isinstance(raw, dict):
        raw = {}
    return {field: numeric_feature(raw.get(field, 0.0)) for field in active_fields("structure")}


def enclosing_subgraph(
    G: nx.Graph,
    u: str,
    v: str,
    snapshot_year: int,
    max_k: int = 4,
    max_p: int = 8,
    structure_fn=None,
) -> Dict[str, Any]:
    paths, _, _ = extract_path.enumerate_induced_paths(G, u, v, L=max_k, max_p=max_p)
    paths = paths or []
    nodes = set()
    for p in paths:
        nodes.update(p)
    G_sub = G.subgraph(nodes).copy() if nodes else nx.Graph()
    edge_list = []
    for a, b in sorted(G_sub.edges()):
        yd = edge_year_dict(G_sub, a, b)
        citations_by_year = {int(y): int(c) for y, c in yd.items()}
        edge_list.append([a, b, citations_by_year])
    return {
        "query_pair": [u, v],
        "edge_list": edge_list,
        "structure_features": compute_structure_features(G, u, v, snapshot_year, structure_fn),
    }


def feature_json_example(agent: str) -> str:
    fields = active_fields(agent)
    body = ",\n".join(f'  "{field}": 0.00' for field in fields)
    return "{\n" + body + "\n}"


def agent_response_format(agent: str) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{agent}_agent_features",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    field: {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    }
                    for field in active_fields(agent)
                },
                "required": list(active_fields(agent)),
                "additionalProperties": False,
            },
        },
    }


def semantic_prompt(ir: int, prediction_window_years: int) -> str:
    return (
        f"""You are the Semantic Agent. Evaluate the semantic feature fields using topical and conceptual compatibility from concept names and local context. The query pair is currently unconnected; do not penalize the lack of a direct edge.

[TASK]
Given a pair of scientific concepts (X, Y), estimate how likely they are to co-occur in papers within the next {prediction_window_years} years with cumulative citations reaching at least {ir}.

[FEATURES]
{feature_descriptions(ACTIVE_FEATURE_SCHEMA, "semantic")}

[OUTPUT]
Return exactly this JSON object with all required keys:
{feature_json_example("semantic")}
Each value must be a JSON number between 0.00 and 1.00. Do not omit keys. Do not add extra keys. Do NOT include explanations, rationales, strings, or any text outside the JSON object.
"""
    )


def trend_prompt(ir: int, prediction_window_years: int) -> str:
    return (
        f"""You are the Trend Agent. Evaluate the trend feature fields using historical temporal signals in the subgraph, including year-by-year citations and recency patterns. The query pair is currently unconnected; do not penalize the lack of a direct edge.

[TASK]
Given a pair of scientific concepts (X, Y), estimate how likely they are to co-occur in papers within the next {prediction_window_years} years with cumulative citations reaching at least {ir}.

[FEATURES]
{feature_descriptions(ACTIVE_FEATURE_SCHEMA, "trend")}

[OUTPUT]
Return exactly this JSON object with all required keys:
{feature_json_example("trend")}
Each value must be a JSON number between 0.00 and 1.00. Do not omit keys. Do not add extra keys. Do NOT include explanations, rationales, strings, or any text outside the JSON object.
"""
    )


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    return round(max(0.0, min(1.0, score)), 2)


def numeric_feature(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    if not np.isfinite(score):
        score = 0.0
    return float(round(score, 6))


def normalize_agent_features(agent: str, value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        data = value
    elif isinstance(value, (int, float)):
        data = {"score": value}
    else:
        data = {}
    if agent == "structure":
        return {field: numeric_feature(data.get(field, 0.0)) for field in active_fields(agent)}
    return {field: clamp_score(data.get(field, 0.0)) for field in active_fields(agent)}


def normalize_feature_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    for agent in AGENTS:
        out[agent] = normalize_agent_features(agent, out.get(agent))
    return out


def has_full_agent_features(record: Dict[str, Any]) -> bool:
    for agent in AGENTS:
        value = record.get(agent)
        if not isinstance(value, dict):
            return False
        if any(field not in value for field in active_fields(agent)):
            return False
    return True


def has_agent_features(record: Dict[str, Any], agents: Tuple[str, ...]) -> bool:
    for agent in agents:
        value = record.get(agent)
        if not isinstance(value, dict):
            return False
        if any(field not in value for field in active_fields(agent)):
            return False
    return True


def run_agent(client: OpenAI, model: str, temp: float, agent: str, sys_prompt: str, query_block: str) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query_block},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temp,
        response_format=agent_response_format(agent),
    )
    content = resp.choices[0].message.content
    data = json.loads(content)
    return normalize_agent_features(agent, data)


def call_multi_agent(
    client: OpenAI,
    model: str,
    temperature: float,
    ir: int,
    prediction_window_years: int,
    concept1: str,
    concept2: str,
    subgraph: Dict[str, Any],
) -> Dict[str, Any]:
    structure_features = normalize_agent_features("structure", subgraph.get("structure_features", {}))
    agent_subgraph = {k: v for k, v in subgraph.items() if k != "structure_features"}
    query_block = (
        f"Concept Pair:\n- Concept X: {concept1}\n- Concept Y: {concept2}\n\n"
        "[SUBGRAPH]\n"
        f"{json.dumps(agent_subgraph, ensure_ascii=False, indent=2)}"
    )
    sem_res = run_agent(client, model, temperature, "semantic", semantic_prompt(ir, prediction_window_years), query_block)
    trend_res = run_agent(client, model, temperature, "trend", trend_prompt(ir, prediction_window_years), query_block)
    return {
        "structure": structure_features,
        "semantic": sem_res,
        "trend": trend_res,
        "concept1": concept1,
        "concept2": concept2,
    }


def save_scores_cache(path: str, records: List[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(normalize_feature_record(r), ensure_ascii=False) + "\n")


def load_scores_cache(path: str, require_full_agent_features: bool = False) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                if require_full_agent_features and not has_full_agent_features(raw):
                    continue
                out.append(normalize_feature_record(raw))
            except Exception:
                continue
    return out


def load_raw_scores_cache(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def has_label(row: Dict[str, Any]) -> bool:
    return "label" in row and not pd.isna(row["label"])


def record_matches_row(record: Dict[str, Any], row: Dict[str, Any]) -> bool:
    return (
        str(record.get("concept1", "")) == str(row.get("concept1", ""))
        and str(record.get("concept2", "")) == str(row.get("concept2", ""))
    )


def cache_matches_rows(records: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> bool:
    if len(records) != len(rows):
        return False
    seen = set()
    for rec in records:
        try:
            idx = int(rec["index"])
        except Exception:
            return False
        if idx in seen or idx < 0 or idx >= len(rows):
            return False
        if not record_matches_row(rec, rows[idx]):
            return False
        seen.add(idx)
    return True


def normalize_dataset_name(dataset: str) -> str:
    name = dataset.upper()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"--dataset must be one of {SUPPORTED_DATASETS}, got {dataset}")
    return name


def resolve_pipeline_paths(args: argparse.Namespace) -> None:
    if args.experiment:
        if not args.dataset:
            raise ValueError("--dataset is required when --experiment is set.")
        dataset = normalize_dataset_name(args.dataset)
        if args.experiment == "test":
            base_dir = os.path.join("dataset", dataset, "test")
            args.train_csv = args.train_csv or os.path.join(base_dir, f"test_train_ir{args.ir}.csv")
            args.test_csv = args.test_csv or os.path.join(base_dir, f"test_test_ir{args.ir}.csv")
            default_train_graph = os.path.join(base_dir, "concept_graph_2023.csv")
            default_test_graph = default_train_graph
        else:
            eval_dir = os.path.join("dataset", dataset, "eval")
            test_dir = os.path.join("dataset", dataset, "test")
            args.train_csv = args.train_csv or os.path.join(eval_dir, f"eval_train_ir{args.ir}.csv")
            args.test_csv = args.test_csv or os.path.join(test_dir, f"test_test_ir{args.ir}.csv")
            default_train_graph = os.path.join(eval_dir, "concept_graph_2021.csv")
            default_test_graph = os.path.join(test_dir, "concept_graph_2023.csv")
        args.train_concept_graph = args.train_concept_graph or args.concept_graph or default_train_graph
        args.test_concept_graph = args.test_concept_graph or args.concept_graph or default_test_graph

        output_dir = args.output_dir or os.path.join("results", f"{dataset.lower()}_{args.experiment}_ir{args.ir}")
        args.scores_cache_train = args.scores_cache_train or os.path.join(output_dir, "train_scores.jsonl")
        args.scores_cache_test = args.scores_cache_test or os.path.join(output_dir, "test_scores.jsonl")

    if args.concept_graph and not args.train_concept_graph:
        args.train_concept_graph = args.concept_graph
    if args.concept_graph and not args.test_concept_graph:
        args.test_concept_graph = args.concept_graph

    missing = [name for name in ("train_csv", "test_csv", "train_concept_graph", "test_concept_graph") if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required path(s): {', '.join('--' + m for m in missing)}")

    args.scores_cache_train = args.scores_cache_train or os.path.join("results", "metallm", "train_scores.jsonl")
    args.scores_cache_test = args.scores_cache_test or os.path.join("results", "metallm", "test_scores.jsonl")


def generate_agent_scores(
    api_key: str,
    model: str,
    temperature: float,
    request_timeout: float,
    api_max_retries: int,
    ir: int,
    prediction_window_years: int,
    rows: List[Dict[str, Any]],
    subgraph_fn,
    max_workers: int,
    keep_label: bool = True,
    cache_path: str = "",
) -> List[Dict[str, Any]]:
    existing_by_index: Dict[int, Dict[str, Any]] = {}
    if cache_path and os.path.exists(cache_path):
        for raw in load_raw_scores_cache(cache_path):
            try:
                idx = int(raw["index"])
            except Exception:
                continue
            if not (0 <= idx < len(rows)) or idx in existing_by_index or not record_matches_row(raw, rows[idx]):
                continue
            if has_full_agent_features(raw):
                existing_by_index[idx] = normalize_feature_record(raw)
                continue
            if has_agent_features(raw, ("semantic", "trend")):
                subg = subgraph_fn(rows[idx]["concept1"], rows[idx]["concept2"])
                migrated = dict(raw)
                migrated["structure"] = subg.get("structure_features", {})
                existing_by_index[idx] = normalize_feature_record(migrated)
    if existing_by_index:
        print(f"[CACHE] Resuming {len(existing_by_index)}/{len(rows)} completed records from {cache_path}")

    outputs: List[Dict[str, Any]] = list(existing_by_index.values())
    completed_indices = set(existing_by_index.keys())
    thread_local = threading.local()
    prepared_rows: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []

    for idx, row in tqdm(list(enumerate(rows)), total=len(rows), desc="Subgraph extraction"):
        if idx in completed_indices:
            continue
        prepared_rows.append((idx, row, subgraph_fn(row["concept1"], row["concept2"])))

    def get_client() -> OpenAI:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=api_key, timeout=request_timeout, max_retries=api_max_retries)
            thread_local.client = client
        return client

    def task(prepared: Tuple[int, Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
        idx, row, subg = prepared
        res = call_multi_agent(
            get_client(),
            model,
            temperature,
            ir,
            prediction_window_years,
            row["concept1"],
            row["concept2"],
            subg,
        )
        if keep_label and has_label(row):
            res["ground_truth_label"] = int(row["label"])
        res["index"] = idx
        return normalize_feature_record(res)

    cache_file = None
    if cache_path:
        parent = os.path.dirname(cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        cache_mode = "a" if completed_indices else "w"
        cache_file = open(cache_path, cache_mode, encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(task, prepared) for prepared in prepared_rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Agent scoring"):
                rec = fut.result()
                outputs.append(rec)
                if cache_file:
                    cache_file.write(json.dumps(normalize_feature_record(rec), ensure_ascii=False) + "\n")
                    cache_file.flush()
    finally:
        if cache_file:
            cache_file.close()
    outputs.sort(key=lambda r: int(r.get("index", 0)))
    return outputs


def parse_args():
    p = argparse.ArgumentParser(description="Multi-agent scoring only (no meta inference).")
    p.add_argument("--experiment", choices=["test", "eval"], default=None, help="Use built-in Test/Eval dataset paths.")
    p.add_argument("--dataset", default=None, help="Dataset name for --experiment.")
    p.add_argument("--train_csv", default=None, help="Training CSV (concept1, concept2, label).")
    p.add_argument("--test_csv", default=None, help="Test/Eval CSV.")
    p.add_argument("--concept_graph", default=None, help="Shared concept graph path for train and test scoring.")
    p.add_argument("--train_concept_graph", default=None, help="Concept graph path for train scoring context.")
    p.add_argument("--test_concept_graph", default=None, help="Concept graph path for test/eval scoring context.")
    p.add_argument("--ir", type=int, required=True, help="IR threshold.")
    p.add_argument("--prediction_window_years", type=int, default=2, help="Future prediction window length used in scoring prompts.")
    p.add_argument("--model", default=DEFAULT_SCORING_MODEL, help="OpenAI model for agent scoring.")
    p.add_argument("--temperature", type=float, default=DEFAULT_SCORING_TEMPERATURE, help="Agent scoring temperature.")
    p.add_argument("--request_timeout", type=float, default=60.0, help="OpenAI request timeout in seconds.")
    p.add_argument("--api_max_retries", type=int, default=2, help="OpenAI client retry count.")
    p.add_argument("--api_key", default=None, help="OpenAI API key.")
    p.add_argument("--feature_schema", required=True, help="JSON schema generated by feature_designer.py.")
    p.add_argument("--structure_code", default=None, help="Python module generated by feature_designer.py for structural features.")
    p.add_argument("--max_workers", type=int, default=100, help="ThreadPoolExecutor workers for LLM calls.")
    p.add_argument("--output_dir", default=None, help="Default output dir when --experiment is used.")
    p.add_argument("--scores_cache_train", type=str, default=None, help="Cache path for train agent scores.")
    p.add_argument("--scores_cache_test", type=str, default=None, help="Cache path for test agent scores.")
    p.add_argument("--reuse_train_cache", action="store_true", help="Reuse train scores cache.")
    p.add_argument("--reuse_test_cache", action="store_true", help="Reuse test scores cache.")
    p.add_argument("--keep_test_labels", action="store_true", help="Attach labels for test rows when available.")
    p.add_argument("--drop_test_labels", action="store_true", help="Do not write test labels even if the CSV has them.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for any shuffling.")
    return p.parse_args()


def resolve_structure_code_path(args: argparse.Namespace, schema: Dict[str, Any]) -> str:
    path = args.structure_code or schema.get("structure_code_path") or ""
    if not path:
        return ""
    if os.path.exists(path):
        return path
    if args.feature_schema:
        alt = os.path.join(os.path.dirname(args.feature_schema), path)
        if os.path.exists(alt):
            return alt
    return path


def main():
    args = parse_args()
    resolve_pipeline_paths(args)
    schema = load_feature_schema(args.feature_schema)
    set_active_feature_schema(schema)
    structure_code_path = resolve_structure_code_path(args, schema)
    if not structure_code_path:
        raise ValueError("Structural feature code is required. Pass --structure_code or set structure_code_path in --feature_schema.")
    structure_fn = load_structure_feature_function(structure_code_path)
    print(
        "[FEATURE_SCHEMA] "
        f"structure={len(active_fields('structure'))}, "
        f"semantic={len(active_fields('semantic'))}, "
        f"trend={len(active_fields('trend'))}"
    )
    print(f"[STRUCTURE_CODE] {structure_code_path}")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")

    G_train, snapshot_year_train = load_concept_graph_csv(args.train_concept_graph)
    G_test, snapshot_year_test = load_concept_graph_csv(args.test_concept_graph)

    def train_subgraph_fn(a: str, b: str) -> Dict[str, Any]:
        return enclosing_subgraph(G_train, a, b, snapshot_year_train, max_k=4, max_p=8, structure_fn=structure_fn)

    def test_subgraph_fn(a: str, b: str) -> Dict[str, Any]:
        return enclosing_subgraph(G_test, a, b, snapshot_year_test, max_k=4, max_p=8, structure_fn=structure_fn)

    np.random.seed(args.seed)

    train_records = None
    if args.reuse_train_cache and os.path.exists(args.scores_cache_train):
        train_rows_for_cache = pd.read_csv(args.train_csv).to_dict(orient="records")
        train_records = load_scores_cache(args.scores_cache_train, require_full_agent_features=True)
        if train_records and cache_matches_rows(train_records, train_rows_for_cache):
            print(f"[CACHE] Reusing train scores from {args.scores_cache_train}")
        else:
            train_records = None
            print("[CACHE] Existing train cache does not match the current rows and active agent schema; regenerating.")
    if train_records is None:
        train_df = pd.read_csv(args.train_csv)
        train_rows = train_df.to_dict(orient="records")
        train_records = generate_agent_scores(
            api_key,
            args.model,
            args.temperature,
            args.request_timeout,
            args.api_max_retries,
            args.ir,
            args.prediction_window_years,
            train_rows,
            train_subgraph_fn,
            args.max_workers,
            keep_label=True,
            cache_path=args.scores_cache_train,
        )
        if args.scores_cache_train:
            save_scores_cache(args.scores_cache_train, train_records)
            print(f"[CACHE] Saved train scores to {args.scores_cache_train}")

    test_records = None
    if args.reuse_test_cache and os.path.exists(args.scores_cache_test):
        test_rows_for_cache = pd.read_csv(args.test_csv).to_dict(orient="records")
        test_records = load_scores_cache(args.scores_cache_test, require_full_agent_features=True)
        if test_records and cache_matches_rows(test_records, test_rows_for_cache):
            print(f"[CACHE] Reusing test scores from {args.scores_cache_test}")
        else:
            test_records = None
            print("[CACHE] Existing test cache does not match the current rows and active agent schema; regenerating.")
    if test_records is None:
        test_df = pd.read_csv(args.test_csv)
        test_rows = test_df.to_dict(orient="records")
        keep_test_label = (args.keep_test_labels or ("label" in test_df.columns)) and not args.drop_test_labels
        test_records = generate_agent_scores(
            api_key,
            args.model,
            args.temperature,
            args.request_timeout,
            args.api_max_retries,
            args.ir,
            args.prediction_window_years,
            test_rows,
            test_subgraph_fn,
            args.max_workers,
            keep_label=keep_test_label,
            cache_path=args.scores_cache_test,
        )
        if args.scores_cache_test:
            save_scores_cache(args.scores_cache_test, test_records)
            print(f"[CACHE] Saved test scores to {args.scores_cache_test}")

    print("[DONE] scores computed and cached.")


if __name__ == "__main__":
    main()
