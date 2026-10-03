import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI

from feature_schema import save_feature_schema, validate_feature_schema
from mirograph_config import DEFAULT_CODE_MODEL, SUPPORTED_DATASETS

load_dotenv(override=True)

def clean_code_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("python"):
            text = text[6:].strip()
    return text


def sanitize_schema(schema: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    validate_feature_schema(schema)
    for agent in ("structure", "semantic", "trend"):
        features = schema["agents"][agent]["features"]
        if not features:
            raise ValueError(f"{agent} feature schema must contain at least one feature.")
        for feature in features:
            feature.setdefault("description", "")
            feature.setdefault("direction", "higher means stronger evidence")
            feature.setdefault("value_range", "numeric")
            feature.setdefault("source", "graph" if agent == "structure" else "llm")
    schema.setdefault("version", 1)
    schema.setdefault("task", "high-impact future scientific concept-pair prediction")
    schema.setdefault("ir", args.ir)
    schema.setdefault("prediction_window_years", args.prediction_window_years)
    return schema


def normalize_dataset_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().upper()
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset {value!r}; expected one of {SUPPORTED_DATASETS}.")
    return normalized


def infer_dataset_from_paths(*paths: Optional[str]) -> Optional[str]:
    supported = {name.upper(): name for name in SUPPORTED_DATASETS}
    for path in paths:
        if not path:
            continue
        parts = re.split(r"[\\/_.-]+", os.path.normpath(path))
        for part in parts:
            match = supported.get(part.upper())
            if match:
                return match
    return None


def split_candidate_csv_args(values: Optional[List[str]]) -> List[str]:
    paths: List[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                paths.append(item)
    return paths


def default_dataset_paths(dataset: str, setting: str, ir: int) -> Tuple[str, List[str]]:
    base = os.path.join("dataset", dataset)
    normalized_setting = setting.lower()
    if normalized_setting == "eval":
        graph_path = os.path.join(base, "eval", "concept_graph_2021.csv")
        candidate_paths = [
            os.path.join(base, "eval", f"eval_train_ir{ir}.csv"),
            os.path.join(base, "test", f"test_test_ir{ir}.csv"),
        ]
    else:
        graph_path = os.path.join(base, "test", "concept_graph_2023.csv")
        candidate_paths = [
            os.path.join(base, "test", f"test_train_ir{ir}.csv"),
            os.path.join(base, "test", f"test_test_ir{ir}.csv"),
        ]
    return graph_path, candidate_paths


def graph_stats_from_csv(csv_path: str) -> Dict[str, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"concept graph CSV not found: {csv_path}")

    nodes = set()
    edges = set()
    max_year: Optional[int] = None
    graph = nx.Graph()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"concept1", "concept2", "year", "citation_count"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path} must have columns {sorted(required)}")
        for row in reader:
            c1 = str(row["concept1"])
            c2 = str(row["concept2"])
            if not c1 or not c2 or c1 == c2:
                continue
            try:
                year = int(row["year"])
            except Exception:
                year = None
            if year is not None:
                max_year = year if max_year is None else max(max_year, year)
            a, b = (c1, c2) if c1 <= c2 else (c2, c1)
            nodes.add(a)
            nodes.add(b)
            edges.add((a, b))

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    average_degree = (2.0 * edge_count / node_count) if node_count else 0.0
    degrees = sorted(dict(graph.degree()).values())
    degree_min = float(degrees[0]) if degrees else 0.0
    degree_max = float(degrees[-1]) if degrees else 0.0
    degree_median = percentile(degrees, 0.50)
    degree_p25 = percentile(degrees, 0.25)
    degree_p75 = percentile(degrees, 0.75)
    degree_std = (
        math.sqrt(sum((degree - average_degree) ** 2 for degree in degrees) / len(degrees))
        if degrees
        else 0.0
    )
    density = nx.density(graph) if node_count > 1 else 0.0
    connected_components = nx.number_connected_components(graph) if node_count else 0
    return {
        "concept_graph": csv_path,
        "snapshot_year": max_year,
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "average_degree": float(average_degree),
        "degree_min": degree_min,
        "degree_p25": degree_p25,
        "degree_median": degree_median,
        "degree_p75": degree_p75,
        "degree_max": degree_max,
        "degree_std": degree_std,
        "density": float(density),
        "connected_components": int(connected_components),
    }


def percentile(sorted_values: List[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    return float(sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac)


def candidate_stats_from_csvs(csv_paths: List[str]) -> Dict[str, Any]:
    total_rows = 0
    unique_pairs = set()
    existing_paths = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"candidate CSV not found: {csv_path}")
        existing_paths.append(csv_path)
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            has_pair_columns = {"concept1", "concept2"}.issubset(fields)
            for row in reader:
                total_rows += 1
                if has_pair_columns:
                    c1 = str(row.get("concept1", ""))
                    c2 = str(row.get("concept2", ""))
                    if c1 and c2:
                        unique_pairs.add((c1, c2) if c1 <= c2 else (c2, c1))
    return {
        "candidate_csvs": existing_paths,
        "candidate_pair_count": int(total_rows),
        "unique_candidate_pair_count": int(len(unique_pairs)) if unique_pairs else int(total_rows),
    }


def build_dataset_stats(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    dataset = normalize_dataset_name(args.dataset) or infer_dataset_from_paths(
        args.output_schema,
        args.output_structure_code,
        args.concept_graph,
        *(args.candidate_csv or []),
    )
    candidate_paths = split_candidate_csv_args(args.candidate_csv)
    graph_path = args.concept_graph

    if dataset and (not graph_path or not candidate_paths):
        default_graph_path, default_candidate_paths = default_dataset_paths(dataset, args.setting, args.ir)
        graph_path = graph_path or default_graph_path
        candidate_paths = candidate_paths or default_candidate_paths

    if not graph_path and not candidate_paths:
        return None
    if not graph_path:
        raise ValueError("--concept_graph is required when --candidate_csv is provided without an inferable --dataset.")
    if not candidate_paths:
        raise ValueError("--candidate_csv is required when --concept_graph is provided without an inferable --dataset.")

    stats: Dict[str, Any] = {
        "dataset": dataset,
        "venue": dataset,
        "setting": args.setting,
    }
    stats.update(graph_stats_from_csv(graph_path))
    stats.update(candidate_stats_from_csvs(candidate_paths))
    return stats


def dataset_context_block(stats: Optional[Dict[str, Any]]) -> str:
    if not stats:
        return ""
    return f"""
[SCG SUMMARY STATISTICS]
- venue: {stats.get("venue") or stats.get("dataset") or "unknown"}
- dataset: {stats.get("dataset") or "unknown"}
- setting: {stats.get("setting") or "unknown"}
- snapshot_year: {stats.get("snapshot_year")}
- graph_size: nodes={stats["node_count"]}, edges={stats["edge_count"]}
- density: {stats["density"]:.8f}
- degree_distribution: min={stats["degree_min"]:.2f}, p25={stats["degree_p25"]:.2f}, mean={stats["average_degree"]:.4f}, median={stats["degree_median"]:.2f}, p75={stats["degree_p75"]:.2f}, max={stats["degree_max"]:.2f}, std={stats["degree_std"]:.4f}
- connected_components: {stats["connected_components"]}
- candidate_pair_count: {stats["candidate_pair_count"]}
- unique_candidate_pair_count: {stats["unique_candidate_pair_count"]}

Use the venue and aggregate SCG statistics to adapt feature schemas to the venue scale and graph sparsity.
Do not define features that require dataset labels, future windows, test labels, or candidate-row totals.

[COMPUTATIONAL CONSTRAINT]
Feature definitions should be practical for repeated evaluation over many candidate pairs in
historical scientific concept graphs.
"""


def schema_prompt(args: argparse.Namespace) -> str:
    dataset_context = dataset_context_block(getattr(args, "dataset_stats", None))
    return f"""You are the Feature Designer Agent for MiroGraph.

MiroGraph predicts whether an currently unconnected pair of scientific concepts will become a high-impact pair in the future prediction window. A positive label means the pair appears in the future and cumulative citations reach IR={args.ir}.
{dataset_context}

Design venue-specific feature schemas for three downstream scoring agents:
- structure: graph-computable structural features over the historical Scientific Concept Graph.
- semantic: LLM-scored semantic features.
- trend: LLM-scored trend features.

Constraints:
- Return a compact, non-empty set of features for each agent.
- Avoid exhaustive feature lists and avoid redundant variants of the same evidence.
- Feature names must be short snake_case.
- Each feature needs a one-sentence description, direction, value_range, and source.
- Structure features must be computable from a NetworkX historical SCG only: nodes are concept strings, undirected edges have citations_by_year={{year: citation_count}}.
- Structure features must be efficient for thousands of candidate pairs. Prefer features whose per-pair computation is local after reusable graph-level preparation.
- The three agents must be mutually disentangled:
  - structure features must not include semantic or trend evidence.
  - semantic features must not include structural or trend evidence.
  - trend features must not include structural or semantic evidence.
- Assign each proposed feature to exactly one agent by the evidence type it uses, and do not duplicate the same evidence across agents under different names.
- Do not use labels, total_citations from dataset rows, future windows, test labels, or paper text unavailable in the graph.
- Semantic and trend features should be usable as [FEATURES] prompt lines for the scoring agents.

Return JSON only in this exact shape:
{{
  "version": 1,
  "task": "high-impact future scientific concept-pair prediction",
  "ir": {args.ir},
  "prediction_window_years": {args.prediction_window_years},
  "agents": {{
    "structure": {{
      "features": [
        {{"name": "feature_name", "description": "one sentence", "direction": "higher means ...", "value_range": "numeric", "source": "graph"}}
      ]
    }},
    "semantic": {{
      "features": [
        {{"name": "feature_name", "description": "one sentence", "direction": "higher means ...", "value_range": "0 to 1", "source": "llm"}}
      ]
    }},
    "trend": {{
      "features": [
        {{"name": "feature_name", "description": "one sentence", "direction": "higher means ...", "value_range": "0 to 1", "source": "llm"}}
      ]
    }}
  }}
}}"""


def generate_feature_schema(client: OpenAI, args: argparse.Namespace) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You design compact, reproducible ML feature schemas. Output JSON only."},
            {"role": "user", "content": schema_prompt(args)},
        ],
        temperature=args.temperature,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    schema = json.loads(content)
    return sanitize_schema(schema, args)


def structure_code_prompt(schema: Dict[str, Any]) -> str:
    structure_features = schema["agents"]["structure"]["features"]
    feature_names = [feature["name"] for feature in structure_features]
    return f'''You are the Structural Feature Code Agent for MiroGraph.

Write a standalone Python module that computes the structural feature schema below from a historical Scientific Concept Graph.

Graph contract:
- G is a networkx.Graph.
- Nodes are scientific concept strings.
- Edges are undirected.
- Each edge may have data["citations_by_year"] as a dict {{year: citation_count}}.
- u and v are the query concepts.
- snapshot_year is the maximum historical year in the graph.

Rules:
- Do not use labels, dataset total_citations, future edges, future years, test labels, files, network, randomness, or OpenAI.
- Implement only the structural schema provided by the Feature Designer Agent.
- Preserve the intended separation between structural, semantic, and trend evidence in the schema.
- Use only Python standard library, math, numpy, and networkx.
- Return numeric finite values for all features.
- Missing nodes should return zeros for all features.
- Keep feature order exactly as FEATURE_NAMES.
- Cache reusable graph-level computations at module level keyed by id(G) and snapshot_year. Do not recompute graph-wide quantities for every query pair.
- Avoid algorithms whose cost scales poorly across thousands of query pairs.
- Keep per-query work local to the query pair and rely on reusable graph-level caches when needed.
- The module must define:
  FEATURE_NAMES = {json.dumps(feature_names, ensure_ascii=False)}
  def compute_structure_features(G, u, v, snapshot_year): ...

Structural schema:
{json.dumps(structure_features, ensure_ascii=False, indent=2)}

Return only executable Python code. Do not wrap in markdown fences.
'''


def validate_structure_code(code: str, schema: Dict[str, Any]) -> None:
    compile(code, "<generated_structural_features>", "exec")
    ns: Dict[str, Any] = {}
    exec(code, ns, ns)
    expected = [feature["name"] for feature in schema["agents"]["structure"]["features"]]
    if ns.get("FEATURE_NAMES") != expected:
        raise ValueError("Generated FEATURE_NAMES does not match the structure schema.")
    fn = ns.get("compute_structure_features")
    if not callable(fn):
        raise ValueError("Generated code must define compute_structure_features.")

    G = nx.Graph()
    G.add_edge("x", "a", citations_by_year={2020: 3, 2021: 5})
    G.add_edge("a", "y", citations_by_year={2021: 7, 2022: 11})
    G.add_edge("x", "b", citations_by_year={2022: 2})
    G.add_edge("b", "y", citations_by_year={2023: 13})
    out = fn(G, "x", "y", 2023)
    if not isinstance(out, dict):
        raise ValueError("compute_structure_features must return a dict.")
    missing = [name for name in expected if name not in out]
    extra = [name for name in out if name not in expected]
    if missing or extra:
        raise ValueError(f"Generated output fields mismatch. missing={missing}, extra={extra}")
    for name in expected:
        value = out[name]
        if not isinstance(value, (int, float)):
            raise ValueError(f"Feature {name} returned non-numeric value {value!r}.")
        if not (-1e100 < float(value) < 1e100):
            raise ValueError(f"Feature {name} returned non-finite value {value!r}.")


def generate_structure_code(client: OpenAI, schema: Dict[str, Any], args: argparse.Namespace) -> str:
    last_error = ""
    for attempt in range(args.code_retries + 1):
        prompt = structure_code_prompt(schema)
        if last_error:
            prompt += f"\nPrevious code failed validation with this error:\n{last_error}\nFix it."
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You write correct, deterministic Python modules. Output only code."},
                {"role": "user", "content": prompt},
            ],
            temperature=args.temperature,
        )
        code = clean_code_text(resp.choices[0].message.content or "")
        try:
            validate_structure_code(code, schema)
            return code
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(f"Could not generate valid structural feature code: {last_error}")


def write_code(path: str, code: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code.rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MiroGraph feature schema and structural feature code.")
    p.add_argument("--output_schema", default="results/feature_designs/feature_schema.json")
    p.add_argument("--output_structure_code", default="results/feature_designs/generated_structural_features.py")
    p.add_argument("--dataset", default=None, help="Optional venue dataset name. Inferred from output paths when possible.")
    p.add_argument(
        "--setting",
        default="test",
        choices=["test", "eval"],
        help="Default path profile used when --dataset is provided without explicit graph/candidate paths.",
    )
    p.add_argument("--concept_graph", default=None, help="Historical concept graph CSV used to summarize dataset scale.")
    p.add_argument(
        "--candidate_csv",
        action="append",
        default=None,
        help="Candidate CSV path for pair-count context. Can be repeated or comma-separated.",
    )
    p.add_argument("--model", default=DEFAULT_CODE_MODEL)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--ir", type=int, default=5)
    p.add_argument("--prediction_window_years", type=int, default=2)
    p.add_argument("--min_structure_features", type=int, default=10)
    p.add_argument("--max_structure_features", type=int, default=100)
    p.add_argument("--min_semantic_features", type=int, default=5)
    p.add_argument("--max_semantic_features", type=int, default=10)
    p.add_argument("--min_trend_features", type=int, default=5)
    p.add_argument("--max_trend_features", type=int, default=10)
    p.add_argument("--code_retries", type=int, default=2)
    p.add_argument("--api_key", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    if args.min_structure_features > args.max_structure_features:
        raise ValueError("min_structure_features must be <= max_structure_features.")
    if args.min_semantic_features > args.max_semantic_features:
        raise ValueError("min_semantic_features must be <= max_semantic_features.")
    if args.min_trend_features > args.max_trend_features:
        raise ValueError("min_trend_features must be <= max_trend_features.")

    args.dataset_stats = build_dataset_stats(args)

    client = OpenAI(api_key=api_key)
    schema = generate_feature_schema(client, args)
    if args.dataset_stats:
        schema["dataset_stats"] = args.dataset_stats
    structure_code = generate_structure_code(client, schema, args)

    schema["structure_code_path"] = args.output_structure_code
    save_feature_schema(args.output_schema, schema)
    write_code(args.output_structure_code, structure_code)

    print(f"[SCHEMA] saved to {args.output_schema}")
    print(f"[STRUCTURE_CODE] saved to {args.output_structure_code}")
    if args.dataset_stats:
        print(
            "[DATASET_STATS] "
            f"dataset={args.dataset_stats.get('dataset')}, "
            f"nodes={args.dataset_stats['node_count']}, "
            f"edges={args.dataset_stats['edge_count']}, "
            f"avg_degree={args.dataset_stats['average_degree']:.4f}, "
            f"degree_min={args.dataset_stats['degree_min']:.2f}, "
            f"degree_median={args.dataset_stats['degree_median']:.2f}, "
            f"degree_max={args.dataset_stats['degree_max']:.2f}, "
            f"degree_std={args.dataset_stats['degree_std']:.4f}, "
            f"density={args.dataset_stats['density']:.8f}, "
            f"components={args.dataset_stats['connected_components']}, "
            f"candidate_pairs={args.dataset_stats['candidate_pair_count']}"
        )
    for agent in ("structure", "semantic", "trend"):
        count = len(schema["agents"][agent]["features"])
        print(f"[{agent}] {count} features")


if __name__ == "__main__":
    main()
