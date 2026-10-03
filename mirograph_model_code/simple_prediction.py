import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

import extract_path

load_dotenv(override=True)

CONCEPT_ONLY_PROMPT_VERSION = "simple_prediction_concept_only_v1"
SUBGRAPH_PROMPT_VERSION = "simple_prediction_subgraph_v1"
PROMPT_VERSION = CONCEPT_ONLY_PROMPT_VERSION


def infer_dataset_setting_from_path(test_file: str) -> Tuple[Optional[str], Optional[str]]:
    path = os.path.abspath(test_file).replace("\\", "/")
    parts = [part for part in path.split("/") if part]
    if "dataset" in parts:
        index = parts.index("dataset")
        if index + 2 < len(parts):
            return parts[index + 1], parts[index + 2]
    if len(parts) >= 2 and parts[-2] in ("eval", "test"):
        return parts[-3] if len(parts) >= 3 else None, parts[-2]
    return None, None


def load_simple_prompt(
    ir: int,
    prediction_window_years: int,
    snapshot_year: int,
    venue: Optional[str] = None,
    include_subgraph: bool = False,
) -> str:
    start_year = snapshot_year + 1
    end_year = snapshot_year + prediction_window_years
    venue_text = f" for the {venue} venue corpus" if venue else ""
    context_text = (
        " Use the provided historical local subgraph as context; edge_list entries are "
        "[concept_a, concept_b, {year: citations}]."
        if include_subgraph
        else " Use only the concept names and task setting provided in the user message."
    )
    return (
        f"Predict if an unconnected scientific concept pair becomes high-impact "
        f"during {start_year}-{end_year}{venue_text}. "
        f"Yes means future co-occurrence AND cumulative citations >= IR={ir}; otherwise No. "
        f"{context_text} "
        'Return only JSON: {"prediction":"Yes"} or {"prediction":"No"}.'
    )


def prediction_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "simple_prediction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "prediction": {"type": "string", "enum": ["Yes", "No"]},
                },
                "required": ["prediction"],
                "additionalProperties": False,
            },
        },
    }


def prompt_version_for(include_subgraph: bool) -> str:
    return SUBGRAPH_PROMPT_VERSION if include_subgraph else CONCEPT_ONLY_PROMPT_VERSION


def baseline_name_for(include_subgraph: bool) -> str:
    return "subgraph_single_llm" if include_subgraph else "concept_only_single_llm"


def load_concept_graph_csv(csv_path: str) -> Tuple[nx.Graph, int]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"concept graph CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"concept1", "concept2", "year", "citation_count"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} must have columns {sorted(required)}")

    df["year"] = df["year"].astype(int)
    df["citation_count"] = df["citation_count"].astype(int)
    snapshot_year = int(df["year"].max())

    agg: Dict[Tuple[str, str, int], int] = {}
    for row in df.itertuples(index=False):
        c1, c2, year, citation_count = row.concept1, row.concept2, int(row.year), int(row.citation_count)
        a, b = (c1, c2) if str(c1) <= str(c2) else (c2, c1)
        key = (a, b, year)
        agg[key] = agg.get(key, 0) + citation_count

    edge_map: Dict[Tuple[str, str], Dict[int, int]] = {}
    for (a, b, year), citation_count in agg.items():
        edge_map.setdefault((a, b), {})
        edge_map[(a, b)][year] = edge_map[(a, b)].get(year, 0) + citation_count

    graph = nx.Graph()
    for (a, b), year_dict in edge_map.items():
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(a, b, citations_by_year=dict(year_dict))
    return graph, snapshot_year


def edge_year_dict(graph: nx.Graph, u: str, v: str) -> Dict[int, int]:
    if not graph.has_edge(u, v):
        return {}
    data = graph.get_edge_data(u, v) or {}
    year_dict = data.get("citations_by_year", {})
    return dict(year_dict) if isinstance(year_dict, dict) else {}


def enclosing_subgraph_context(
    graph: nx.Graph,
    u: str,
    v: str,
    max_k: int = 4,
    max_p: int = 8,
) -> Dict[str, Any]:
    paths, _, _ = extract_path.enumerate_induced_paths(graph, u, v, L=max_k, max_p=max_p)
    paths = paths or []
    nodes = set()
    for path in paths:
        nodes.update(path)
    subgraph = graph.subgraph(nodes).copy() if nodes else nx.Graph()
    edge_list = []
    for a, b in sorted(subgraph.edges()):
        citations_by_year = {int(year): int(count) for year, count in edge_year_dict(subgraph, a, b).items()}
        edge_list.append([a, b, citations_by_year])
    return {
        "query_pair": [u, v],
        "edge_list": edge_list,
    }


def infer_concept_graph_path(test_file: str, snapshot_year: int) -> Optional[str]:
    dataset, setting = infer_dataset_setting_from_path(test_file)
    test_dir = os.path.dirname(os.path.abspath(test_file))
    candidates = [
        os.path.join(test_dir, f"concept_graph_{snapshot_year}.csv"),
        os.path.join(test_dir, "concept_graph.csv"),
    ]
    if dataset:
        if setting:
            candidates.extend(
                [
                    os.path.join("dataset", dataset, setting, f"concept_graph_{snapshot_year}.csv"),
                    os.path.join("dataset", dataset, setting, "concept_graph.csv"),
                ]
            )
        candidates.extend(
            [
                os.path.join("dataset", dataset, "test", f"concept_graph_{snapshot_year}.csv"),
                os.path.join("dataset", dataset, "test", "concept_graph.csv"),
                os.path.join("dataset", dataset, "eval", f"concept_graph_{snapshot_year}.csv"),
                os.path.join("dataset", dataset, "eval", "concept_graph.csv"),
            ]
        )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def resolve_concept_graph_path(test_file: str, snapshot_year: int, explicit_path: Optional[str]) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"--concept_graph not found: {explicit_path}")
        return explicit_path
    inferred = infer_concept_graph_path(test_file, snapshot_year)
    if inferred:
        return inferred
    raise FileNotFoundError(
        "Could not infer a concept graph path. Pass --concept_graph, e.g. "
        f"dataset/<DATASET>/test/concept_graph_{snapshot_year}.csv"
    )


def build_user_input(concept1: str, concept2: str, subgraph: Optional[Dict[str, Any]] = None) -> str:
    if subgraph is None:
        return f"X: {concept1}\nY: {concept2}\nPredict Yes/No."
    return (
        f"Concept Pair:\n- Concept X: {concept1}\n- Concept Y: {concept2}\n\n"
        "[SUBGRAPH]\n"
        f"{json.dumps(subgraph, ensure_ascii=False, indent=2)}"
    )


def predict_pair(
    client: OpenAI,
    concept1: str,
    concept2: str,
    ir: int,
    model: str = "gpt-4o-mini",
    prediction_window_years: int = 2,
    snapshot_year: int = 2023,
    venue: Optional[str] = None,
    subgraph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    include_subgraph = subgraph is not None
    prompt = load_simple_prompt(ir, prediction_window_years, snapshot_year, venue, include_subgraph)
    user_input = build_user_input(concept1, concept2, subgraph)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ]

    temp = 1.0 if "gpt-5" in model else 0.0
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        response_format=prediction_response_format(),
    )
    result = json.loads(response.choices[0].message.content or "{}")
    prediction = "Yes" if str(result.get("prediction", "")).strip().lower() == "yes" else "No"
    output = {
        "concept1": concept1,
        "concept2": concept2,
        "ir": ir,
        "prediction_window_years": prediction_window_years,
        "snapshot_year": snapshot_year,
        "venue": venue,
        "include_subgraph": include_subgraph,
        "prediction": prediction,
        "prediction_int": 1 if prediction == "Yes" else 0,
    }
    if subgraph is not None:
        nodes = set(subgraph.get("query_pair", []))
        nodes.update({node for edge in subgraph.get("edge_list", []) for node in edge[:2]})
        output["subgraph_edge_count"] = len(subgraph.get("edge_list", []))
        output["subgraph_node_count"] = len(nodes)
    return output


def process_single_pair(args_tuple: Tuple[Any, ...]) -> Tuple[int, Dict[str, Any]]:
    (
        idx,
        row,
        ir,
        model,
        client,
        prediction_window_years,
        snapshot_year,
        venue,
        subgraph,
    ) = args_tuple

    concept1 = str(row["concept1"])
    concept2 = str(row["concept2"])
    ground_truth = int(row["label"])
    total_citations = int(row.get("total_citations", 0))

    result = predict_pair(
        client,
        concept1,
        concept2,
        ir,
        model,
        prediction_window_years=prediction_window_years,
        snapshot_year=snapshot_year,
        venue=venue,
        subgraph=subgraph,
    )
    result["ground_truth_label"] = ground_truth
    result["total_citations"] = total_citations
    result["pair"] = {"concept1": concept1, "concept2": concept2}
    return idx, result


def output_path_for(
    test_file: str,
    ir: int,
    model: str,
    prediction_window_years: int,
    snapshot_year: int,
    include_subgraph: bool = False,
) -> str:
    dataset, setting = infer_dataset_setting_from_path(test_file)
    model_name = model.replace("/", "_")
    baseline = "simple_subgraph" if include_subgraph else "simple_concept_only"
    filename = f"{baseline}_ir{ir}_{model_name}_win{prediction_window_years}_snap{snapshot_year}.jsonl"
    if dataset and setting:
        return os.path.join("results", dataset, setting, filename)
    return os.path.join("results", filename)


def run_batch_prediction(
    client: OpenAI,
    test_file: str,
    ir: int,
    model: str = "gpt-4o-mini",
    output_file: Optional[str] = None,
    batch_size: int = 100,
    prediction_window_years: int = 2,
    snapshot_year: int = 2023,
    venue: Optional[str] = None,
    include_subgraph: bool = False,
    concept_graph_path: Optional[str] = None,
    subgraph_max_k: int = 4,
    subgraph_max_p: int = 8,
) -> Tuple[List[Dict[str, Any]], float, float]:
    df = pd.read_csv(test_file)
    dataset, _setting = infer_dataset_setting_from_path(test_file)
    venue_name = venue or dataset

    if output_file is None:
        output_file = output_path_for(test_file, ir, model, prediction_window_years, snapshot_year, include_subgraph)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    processed_pairs = set()
    all_results: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    concept_graph_used: Optional[str] = None
    graph: Optional[nx.Graph] = None

    if include_subgraph:
        concept_graph_used = resolve_concept_graph_path(test_file, snapshot_year, concept_graph_path)
        graph, graph_snapshot_year = load_concept_graph_csv(concept_graph_used)
        print(f"Loaded concept graph: {concept_graph_used}")
        print(f"Graph snapshot year: {graph_snapshot_year}")

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if data.get("type") != "result":
                    continue
                pair_key = tuple(sorted((str(data.get("concept1")), str(data.get("concept2")))))
                processed_pairs.add(pair_key)
                all_results.append(data)
                y_true.append(int(data["ground_truth_label"]))
                y_pred.append(int(data["prediction_int"]))
        if processed_pairs:
            print(f"Found {len(processed_pairs)} existing results in {output_file}. Resuming...")

    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        metadata = {
            "type": "metadata",
            "prompt_version": prompt_version_for(include_subgraph),
            "baseline": baseline_name_for(include_subgraph),
            "ir": ir,
            "model": model,
            "test_file": test_file,
            "venue": venue_name,
            "prediction_window_years": prediction_window_years,
            "snapshot_year": snapshot_year,
            "include_subgraph": include_subgraph,
            "concept_graph": concept_graph_used,
            "subgraph_max_k": subgraph_max_k if include_subgraph else None,
            "subgraph_max_p": subgraph_max_p if include_subgraph else None,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print(f"Loaded {len(df)} pairs from {test_file}")
    if venue_name:
        print(f"Venue corpus: {venue_name}")
    print(f"Prediction window: {snapshot_year + 1}-{snapshot_year + prediction_window_years}")
    print(f"IR: {ir}")
    print(f"Model: {model}")
    print(f"Subgraph context: {'enabled' if include_subgraph else 'disabled'}")
    if include_subgraph:
        print(f"Subgraph extraction: max_k={subgraph_max_k}, max_p={subgraph_max_p}")
    print(f"Concurrent workers: {batch_size}")

    args_list = []
    pending_rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        concept1 = str(row["concept1"])
        concept2 = str(row["concept2"])
        if tuple(sorted((concept1, concept2))) in processed_pairs:
            continue
        pending_rows.append((idx, row, concept1, concept2))

    subgraphs: Dict[int, Optional[Dict[str, Any]]] = {}
    if include_subgraph:
        if graph is None:
            raise RuntimeError("Subgraph context requested but concept graph was not loaded.")
        for idx, _row, concept1, concept2 in tqdm(pending_rows, total=len(pending_rows), desc="Subgraph extraction"):
            subgraphs[idx] = enclosing_subgraph_context(graph, concept1, concept2, max_k=subgraph_max_k, max_p=subgraph_max_p)

    for idx, row, _concept1, _concept2 in pending_rows:
        args_list.append(
            (
                idx,
                row,
                ir,
                model,
                client,
                prediction_window_years,
                snapshot_year,
                venue_name,
                subgraphs.get(idx),
            )
        )

    if args_list:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(process_single_pair, args): args[0] for args in args_list}
            with tqdm(total=len(args_list), desc="Predicting") as pbar:
                for future in as_completed(futures):
                    try:
                        _, result = future.result()
                        result["type"] = "result"
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        all_results.append(result)
                        y_true.append(int(result["ground_truth_label"]))
                        y_pred.append(int(result["prediction_int"]))
                    except Exception as exc:
                        print(f"\n[ERROR] Future failed: {exc}")
                    pbar.update(1)
    else:
        print("All pairs already processed.")

    if not y_true:
        print("No results to calculate metrics.")
        return [], 0.0, 0.0

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print()
    print("=" * 80)
    print("Cumulative Results")
    print("=" * 80)
    print(f"Total pairs: {len(all_results)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"TP: {cm[1][1]}, TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}")

    final_metrics = {
        "type": "metrics",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": {
            "tp": int(cm[1][1]),
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
        },
    }
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(final_metrics, ensure_ascii=False) + "\n")
    print(f"[Finished] Saved to: {output_file}")

    return all_results, accuracy, macro_f1


def get_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    api_key_name = f"OPENAI_API_KEY_{args.api_key_name}"
    api_key = os.getenv(api_key_name) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"API key not found. Set {api_key_name} or OPENAI_API_KEY in the environment")
    return api_key


def main() -> Optional[Dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Single-LLM impact prediction baseline")
    parser.add_argument("--concept1", type=str, default=None, help="First concept for single prediction")
    parser.add_argument("--concept2", type=str, default=None, help="Second concept for single prediction")
    parser.add_argument("--test_file", type=str, default=None, help="CSV file for batch prediction")
    parser.add_argument("--ir", type=int, required=True, help="IR threshold")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--api_key_name", type=str, default="1", help="API key suffix, e.g. OPENAI_API_KEY_1")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL/JSON path")
    parser.add_argument("--batch_size", type=int, default=100, help="Concurrent workers for batch prediction")
    parser.add_argument("--prediction_window_years", type=int, default=2, help="Prediction window length")
    parser.add_argument("--snapshot_year", type=int, default=2023, help="Historical cutoff/snapshot year")
    parser.add_argument("--venue", type=str, default=None, help="Venue/corpus name, e.g. CIKM, KDD, SIGIR")
    parser.add_argument("--include_subgraph", action="store_true", help="Provide the same enclosing subgraph context used by MiroGraph semantic/trend agents.")
    parser.add_argument("--concept_graph", type=str, default=None, help="Concept graph CSV for --include_subgraph. Inferred from --test_file when omitted.")
    parser.add_argument("--subgraph_max_k", type=int, default=4, help="Maximum induced path length for subgraph context.")
    parser.add_argument("--subgraph_max_p", type=int, default=8, help="Maximum induced paths for subgraph context.")
    args = parser.parse_args()

    client = OpenAI(api_key=get_api_key(args))

    if args.concept1 and args.concept2:
        subgraph = None
        if args.include_subgraph:
            if not args.concept_graph:
                raise ValueError("--concept_graph is required with --include_subgraph for single-pair prediction.")
            graph, _graph_snapshot_year = load_concept_graph_csv(args.concept_graph)
            subgraph = enclosing_subgraph_context(
                graph,
                args.concept1,
                args.concept2,
                max_k=args.subgraph_max_k,
                max_p=args.subgraph_max_p,
            )
        result = predict_pair(
            client,
            args.concept1,
            args.concept2,
            args.ir,
            args.model,
            prediction_window_years=args.prediction_window_years,
            snapshot_year=args.snapshot_year,
            venue=args.venue,
            subgraph=subgraph,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    if args.test_file:
        run_batch_prediction(
            client=client,
            test_file=args.test_file,
            ir=args.ir,
            model=args.model,
            output_file=args.output,
            batch_size=args.batch_size,
            prediction_window_years=args.prediction_window_years,
            snapshot_year=args.snapshot_year,
            venue=args.venue,
            include_subgraph=args.include_subgraph,
            concept_graph_path=args.concept_graph,
            subgraph_max_k=args.subgraph_max_k,
            subgraph_max_p=args.subgraph_max_p,
        )
        return None

    parser.error("Either --concept1/--concept2 or --test_file must be provided")
    return None


if __name__ == "__main__":
    main()
