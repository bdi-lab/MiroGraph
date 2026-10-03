import argparse
import json
import os
import time
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
from mirograph_config import DEFAULT_SCORING_MODEL, DEFAULT_SCORING_TEMPERATURE, SUPPORTED_DATASETS


load_dotenv(override=True)


def load_concept_graph_csv(csv_path: str) -> Tuple[nx.Graph, int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"concept graph not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"concept1", "concept2", "year", "citation_count"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} must have columns {sorted(required)}")
    df["year"] = df["year"].astype(int)
    df["citation_count"] = df["citation_count"].astype(int)
    snapshot_year = int(df["year"].max())
    edge_map: Dict[Tuple[str, str], Dict[int, int]] = {}
    for row in df.itertuples(index=False):
        c1, c2, year, citation_count = row.concept1, row.concept2, int(row.year), int(row.citation_count)
        a, b = (c1, c2) if str(c1) <= str(c2) else (c2, c1)
        edge_map.setdefault((a, b), {})
        edge_map[(a, b)][year] = edge_map[(a, b)].get(year, 0) + citation_count
    graph = nx.Graph()
    for (a, b), citations_by_year in edge_map.items():
        graph.add_edge(a, b, citations_by_year=dict(citations_by_year))
    return graph, snapshot_year


def edge_year_dict(graph: nx.Graph, u: str, v: str) -> Dict[int, int]:
    if not graph.has_edge(u, v):
        return {}
    data = graph.get_edge_data(u, v) or {}
    value = data.get("citations_by_year", {})
    return dict(value) if isinstance(value, dict) else {}


def enclosing_subgraph(graph: nx.Graph, u: str, v: str, max_k: int = 4, max_p: int = 8) -> Dict[str, Any]:
    paths, _, _ = extract_path.enumerate_induced_paths(graph, u, v, L=max_k, max_p=max_p)
    nodes = set()
    for path in paths or []:
        nodes.update(path)
    subgraph = graph.subgraph(nodes).copy() if nodes else nx.Graph()
    edge_list = []
    for a, b in sorted(subgraph.edges()):
        citations_by_year = {int(y): int(c) for y, c in edge_year_dict(subgraph, a, b).items()}
        edge_list.append([a, b, citations_by_year])
    return {"query_pair": [u, v], "edge_list": edge_list}


def score_prompt(agent: str, ir: int, prediction_window_years: int) -> str:
    if agent == "structure":
        focus = "structural evidence in the historical subgraph, including paths and connectivity"
    elif agent == "semantic":
        focus = "topical and conceptual compatibility from concept names and local context"
    elif agent == "trend":
        focus = "historical temporal signals in the subgraph, including year-by-year citations and recency patterns"
    else:
        raise ValueError(f"unknown agent: {agent}")
    return (
        f"You are the {agent.title()} Agent. Estimate the score using {focus}. "
        "The query pair is currently unconnected; do not penalize the lack of a direct edge.\n\n"
        "[TASK]\n"
        f"Estimate how likely the concept pair is to co-occur within the next {prediction_window_years} years "
        f"with cumulative citations reaching at least IR={ir}.\n\n"
        "[OUTPUT]\n"
        'Return JSON only: {"score": <number between 0 and 1>}. '
        "Do not include reasoning or any extra keys."
    )


def response_format(agent: str) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{agent}_single_score",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["score"],
                "additionalProperties": False,
            },
        },
    }


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    if not np.isfinite(score):
        score = 0.0
    return round(max(0.0, min(1.0, score)), 2)


def run_agent(
    client: OpenAI,
    model: str,
    temperature: float,
    agent: str,
    ir: int,
    prediction_window_years: int,
    query_block: str,
) -> Dict[str, float]:
    last_error = None
    for attempt in range(8):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": score_prompt(agent, ir, prediction_window_years)},
                    {"role": "user", "content": query_block},
                ],
                temperature=temperature,
                response_format=response_format(agent),
            )
            data = json.loads(response.choices[0].message.content)
            return {"score": clamp_score(data.get("score", 0.0))}
        except Exception as exc:
            last_error = exc
            text = str(exc).lower()
            if "rate_limit" not in text and "rate limit" not in text and "429" not in text:
                raise
            time.sleep(min(60.0, 2.0 * (attempt + 1)))
    raise last_error


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
    query_block = (
        f"Concept Pair:\n- Concept X: {concept1}\n- Concept Y: {concept2}\n\n"
        "[SUBGRAPH]\n"
        f"{json.dumps(subgraph, ensure_ascii=False, indent=2)}"
    )
    return {
        "structure": run_agent(client, model, temperature, "structure", ir, prediction_window_years, query_block),
        "semantic": run_agent(client, model, temperature, "semantic", ir, prediction_window_years, query_block),
        "trend": run_agent(client, model, temperature, "trend", ir, prediction_window_years, query_block),
        "concept1": concept1,
        "concept2": concept2,
    }


def load_raw_cache(path: str) -> List[Dict[str, Any]]:
    records = []
    if not path or not os.path.exists(path):
        return records
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


def record_matches_row(record: Dict[str, Any], row: Dict[str, Any]) -> bool:
    return str(record.get("concept1", "")) == str(row.get("concept1", "")) and str(record.get("concept2", "")) == str(row.get("concept2", ""))


def has_score_only(record: Dict[str, Any]) -> bool:
    for agent in ("structure", "semantic", "trend"):
        value = record.get(agent)
        if not isinstance(value, dict) or "score" not in value:
            return False
    return True


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    for agent in ("structure", "semantic", "trend"):
        value = out.get(agent, {})
        if not isinstance(value, dict):
            value = {}
        out[agent] = {"score": clamp_score(value.get("score", 0.0))}
    return out


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
    keep_label: bool,
    cache_path: str,
) -> List[Dict[str, Any]]:
    existing_by_index: Dict[int, Dict[str, Any]] = {}
    for raw in load_raw_cache(cache_path):
        try:
            idx = int(raw["index"])
        except Exception:
            continue
        if 0 <= idx < len(rows) and idx not in existing_by_index and record_matches_row(raw, rows[idx]) and has_score_only(raw):
            existing_by_index[idx] = normalize_record(raw)
    if existing_by_index:
        print(f"[CACHE] Resuming {len(existing_by_index)}/{len(rows)} completed records from {cache_path}")

    prepared_rows = []
    for idx, row in tqdm(list(enumerate(rows)), total=len(rows), desc="Subgraph extraction"):
        if idx not in existing_by_index:
            prepared_rows.append((idx, row, subgraph_fn(row["concept1"], row["concept2"])))

    outputs = list(existing_by_index.values())
    thread_local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAI(api_key=api_key, timeout=request_timeout, max_retries=api_max_retries)
            thread_local.client = client
        return client

    def task(item: Tuple[int, Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
        idx, row, subgraph = item
        result = call_multi_agent(
            get_client(),
            model,
            temperature,
            ir,
            prediction_window_years,
            row["concept1"],
            row["concept2"],
            subgraph,
        )
        if keep_label and "label" in row and not pd.isna(row["label"]):
            result["ground_truth_label"] = int(row["label"])
        result["index"] = idx
        return normalize_record(result)

    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cache_file = open(cache_path, "a" if existing_by_index else "w", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task, row) for row in prepared_rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Agent scoring"):
                record = future.result()
                outputs.append(record)
                cache_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                cache_file.flush()
    finally:
        cache_file.close()
    outputs.sort(key=lambda r: int(r.get("index", 0)))
    return outputs


def save_cache(path: str, records: List[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(normalize_record(record), ensure_ascii=False) + "\n")


def normalize_dataset_name(dataset: str) -> str:
    name = dataset.upper()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"--dataset must be one of {SUPPORTED_DATASETS}, got {dataset}")
    return name


def resolve_paths(args: argparse.Namespace) -> None:
    if args.experiment:
        dataset = normalize_dataset_name(args.dataset)
        if args.experiment == "test":
            base_dir = os.path.join("dataset", dataset, "test")
            args.train_csv = args.train_csv or os.path.join(base_dir, f"test_train_ir{args.ir}.csv")
            args.test_csv = args.test_csv or os.path.join(base_dir, f"test_test_ir{args.ir}.csv")
            args.train_concept_graph = args.train_concept_graph or os.path.join(base_dir, "concept_graph_2023.csv")
            args.test_concept_graph = args.test_concept_graph or args.train_concept_graph
        else:
            train_dir = os.path.join("dataset", dataset, "eval")
            test_dir = os.path.join("dataset", dataset, "test")
            args.train_csv = args.train_csv or os.path.join(train_dir, f"eval_train_ir{args.ir}.csv")
            args.test_csv = args.test_csv or os.path.join(test_dir, f"test_test_ir{args.ir}.csv")
            args.train_concept_graph = args.train_concept_graph or os.path.join(train_dir, "concept_graph_2021.csv")
            args.test_concept_graph = args.test_concept_graph or os.path.join(test_dir, "concept_graph_2023.csv")
        output_dir = args.output_dir or os.path.join("results", f"{dataset.lower()}_{args.experiment}_ir{args.ir}_single_score_gpt4o_mini")
        args.scores_cache_train = args.scores_cache_train or os.path.join(output_dir, "train_scores.jsonl")
        args.scores_cache_test = args.scores_cache_test or os.path.join(output_dir, "test_scores.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-score multi-agent scoring.")
    parser.add_argument("--experiment", choices=["test", "eval"], default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--train_concept_graph", default=None)
    parser.add_argument("--test_concept_graph", default=None)
    parser.add_argument("--ir", type=int, required=True)
    parser.add_argument("--prediction_window_years", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_SCORING_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_SCORING_TEMPERATURE)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--api_max_retries", type=int, default=2)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--scores_cache_train", default=None)
    parser.add_argument("--scores_cache_test", default=None)
    parser.add_argument("--reuse_train_cache", action="store_true")
    parser.add_argument("--reuse_test_cache", action="store_true")
    parser.add_argument("--keep_test_labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment and not args.dataset:
        raise ValueError("--dataset is required with --experiment.")
    resolve_paths(args)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")

    train_graph, _ = load_concept_graph_csv(args.train_concept_graph)
    test_graph, _ = load_concept_graph_csv(args.test_concept_graph)

    train_rows = pd.read_csv(args.train_csv).to_dict(orient="records")
    test_rows = pd.read_csv(args.test_csv).to_dict(orient="records")
    print(f"[INFO] model={args.model} ir={args.ir} train_rows={len(train_rows)} test_rows={len(test_rows)}")
    print(f"[INFO] train_graph={args.train_concept_graph}")
    print(f"[INFO] test_graph={args.test_concept_graph}")

    train_records = None
    if args.reuse_train_cache and args.scores_cache_train and os.path.exists(args.scores_cache_train):
        cached = load_raw_cache(args.scores_cache_train)
        if len(cached) == len(train_rows):
            train_records = [normalize_record(r) for r in cached]
            print(f"[CACHE] Reusing train scores from {args.scores_cache_train}")
    if train_records is None:
        train_records = generate_agent_scores(
            api_key,
            args.model,
            args.temperature,
            args.request_timeout,
            args.api_max_retries,
            args.ir,
            args.prediction_window_years,
            train_rows,
            lambda a, b: enclosing_subgraph(train_graph, a, b),
            args.max_workers,
            keep_label=True,
            cache_path=args.scores_cache_train,
        )
        save_cache(args.scores_cache_train, train_records)
        print(f"[CACHE] Saved train scores to {args.scores_cache_train}")

    test_records = None
    if args.reuse_test_cache and args.scores_cache_test and os.path.exists(args.scores_cache_test):
        cached = load_raw_cache(args.scores_cache_test)
        if len(cached) == len(test_rows):
            test_records = [normalize_record(r) for r in cached]
            print(f"[CACHE] Reusing test scores from {args.scores_cache_test}")
    if test_records is None:
        test_records = generate_agent_scores(
            api_key,
            args.model,
            args.temperature,
            args.request_timeout,
            args.api_max_retries,
            args.ir,
            args.prediction_window_years,
            test_rows,
            lambda a, b: enclosing_subgraph(test_graph, a, b),
            args.max_workers,
            keep_label=args.keep_test_labels,
            cache_path=args.scores_cache_test,
        )
        save_cache(args.scores_cache_test, test_records)
        print(f"[CACHE] Saved test scores to {args.scores_cache_test}")

    print("[DONE] single-score caches are ready.")


if __name__ == "__main__":
    main()
