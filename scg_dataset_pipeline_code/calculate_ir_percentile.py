import argparse
import json
from pathlib import Path
from itertools import combinations
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd


Edge = Tuple[str, str]


def sorted_edge(a: str, b: str) -> Edge:
    return tuple(sorted((str(a).strip(), str(b).strip())))


def parse_concepts(value) -> list:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_graph_edges(path: Path) -> Tuple[Set[Edge], Set[str]]:
    df = pd.read_csv(path, usecols=["concept1", "concept2"])
    edges = set()
    concepts = set()
    for c1, c2 in zip(df["concept1"], df["concept2"]):
        edge = sorted_edge(c1, c2)
        if edge[0] and edge[1] and edge[0] != edge[1]:
            edges.add(edge)
            concepts.update(edge)
    return edges, concepts


def load_graph_edge_totals(path: Path, start_year: int, end_year: int) -> Dict[Edge, int]:
    df = pd.read_csv(path, usecols=["concept1", "concept2", "year", "citation_count"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0).astype(int)
    df = df[df["year"].notna() & (df["year"] >= start_year) & (df["year"] <= end_year)]

    totals: Dict[Edge, int] = {}
    for c1, c2, citation_count in zip(df["concept1"], df["concept2"], df["citation_count"]):
        edge = sorted_edge(c1, c2)
        if edge[0] and edge[1] and edge[0] != edge[1]:
            totals[edge] = totals.get(edge, 0) + int(citation_count)
    return totals


def load_future_edge_totals(path: Path) -> Dict[Edge, int]:
    df = pd.read_csv(path, usecols=["concept1", "concept2", "total_citations"])
    df["total_citations"] = pd.to_numeric(df["total_citations"], errors="coerce").fillna(0).astype(int)
    totals: Dict[Edge, int] = {}
    for c1, c2, total in zip(df["concept1"], df["concept2"], df["total_citations"]):
        edge = sorted_edge(c1, c2)
        if edge[0] and edge[1] and edge[0] != edge[1]:
            totals[edge] = totals.get(edge, 0) + int(total)
    return totals


def cache_record_to_counts(record, paper_year: Optional[int], start_year: int, end_year: int) -> Optional[Dict[int, int]]:
    if record is None:
        return None
    if isinstance(record, dict) and "citation_counts_by_year" in record:
        raw_counts = record.get("citation_counts_by_year", {}) or {}
    elif isinstance(record, dict):
        raw_counts = record
    else:
        return None

    counts: Dict[int, int] = {}
    for year, count in raw_counts.items():
        try:
            y = int(year)
            c = int(count)
        except (TypeError, ValueError):
            continue
        if paper_year is not None and y < int(paper_year):
            continue
        if start_year <= y <= end_year:
            counts[y] = c
    return counts


def calculate_from_final_csv(
    final_csv: Path,
    history_graph: Path,
    cache_file: Path,
    future_start_year: int,
    future_end_year: int,
    citation_start_year: int,
    citation_end_year: int,
    ir: int,
) -> dict:
    history_edges, history_concepts = load_graph_edges(history_graph)
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file else {}

    df = pd.read_csv(final_csv)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[
        df["year"].notna()
        & (df["year"] >= future_start_year)
        & (df["year"] <= future_end_year)
    ].copy()

    edge_to_papers = {}
    paper_totals: Dict[int, int] = {}
    papers_with_candidate_concepts = 0
    cache_hits = 0

    for idx, row in df.iterrows():
        concepts = sorted(set(c for c in parse_concepts(row.get("assigned_concepts", "")) if c in history_concepts))
        if len(concepts) < 2:
            continue
        papers_with_candidate_concepts += 1
        paper_year = int(row["year"])
        counts = cache_record_to_counts(
            cache.get(str(idx)),
            paper_year,
            citation_start_year,
            citation_end_year,
        )
        if counts is not None:
            cache_hits += 1
        paper_totals[int(idx)] = sum(int(v) for v in (counts or {}).values())

        for c1, c2 in combinations(concepts, 2):
            edge = sorted_edge(c1, c2)
            if edge not in history_edges:
                edge_to_papers.setdefault(edge, []).append(int(idx))

    edge_totals = {
        edge: sum(paper_totals.get(paper_id, 0) for paper_id in paper_ids)
        for edge, paper_ids in edge_to_papers.items()
    }
    out = summarize(edge_totals, ir)
    out.update(
        {
            "final_csv": str(final_csv),
            "history_graph": str(history_graph),
            "cache_file": str(cache_file),
            "future_years": [future_start_year, future_end_year],
            "citation_years": [citation_start_year, citation_end_year],
            "future_papers_with_candidate_concepts": papers_with_candidate_concepts,
            "citation_cache_hits": cache_hits,
            "citation_cache_hit_percent": (
                cache_hits / papers_with_candidate_concepts * 100.0 if papers_with_candidate_concepts else 0.0
            ),
        }
    )
    return out


def summarize(edge_totals: Dict[Edge, int], ir: int) -> dict:
    values = list(edge_totals.values())
    denominator = len(values)
    positives = sum(1 for value in values if value >= ir)
    zero = sum(1 for value in values if value == 0)
    below_nonzero = sum(1 for value in values if 0 < value < ir)
    return {
        "future_connected_pairs": denominator,
        "ir": ir,
        "pairs_with_citations_gte_ir": positives,
        "top_percent": (positives / denominator * 100.0) if denominator else 0.0,
        "zero_citation_pairs": zero,
        "zero_citation_percent": (zero / denominator * 100.0) if denominator else 0.0,
        "nonzero_below_ir_pairs": below_nonzero,
        "nonzero_below_ir_percent": (below_nonzero / denominator * 100.0) if denominator else 0.0,
    }


def calculate_from_snapshots(
    history_graph: Path,
    future_graph: Path,
    start_year: int,
    end_year: int,
    ir: int,
) -> dict:
    history_edges, history_concepts = load_graph_edges(history_graph)
    future_edges, _ = load_graph_edges(future_graph)
    future_totals = load_graph_edge_totals(future_graph, start_year, end_year)

    new_edges = {
        edge
        for edge in future_edges
        if edge not in history_edges and edge[0] in history_concepts and edge[1] in history_concepts
    }
    edge_totals = {edge: int(future_totals.get(edge, 0)) for edge in new_edges}
    out = summarize(edge_totals, ir)
    out.update(
        {
            "history_graph": str(history_graph),
            "future_graph": str(future_graph),
            "target_years": [start_year, end_year],
        }
    )
    return out


def calculate_from_future_totals(future_edge_totals: Path, ir: int) -> dict:
    edge_totals = load_future_edge_totals(future_edge_totals)
    out = summarize(edge_totals, ir)
    out["future_edge_totals"] = str(future_edge_totals)
    return out


def default_snapshot_jobs(dataset_root: Path, venues: Iterable[str], ir: int, setting: str) -> list:
    jobs = []
    for venue in venues:
        base = dataset_root / venue
        if setting in {"eval", "both"}:
            jobs.append(
                {
                    "venue": venue,
                    "setting": "eval",
                    "history_graph": base / "eval" / "concept_graph_2021.csv",
                    "future_graph": base / "eval" / "concept_graph_2023.csv",
                    "start_year": 2022,
                    "end_year": 2023,
                    "ir": ir,
                }
            )
        if setting in {"test", "both"}:
            jobs.append(
                {
                    "venue": venue,
                    "setting": "test",
                    "history_graph": base / "test" / "concept_graph_2023.csv",
                    "future_graph": base / "test" / "concept_graph_2025.csv",
                    "start_year": 2024,
                    "end_year": 2025,
                    "ir": ir,
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate what upper percentage an IR threshold represents among future-connected concept pairs."
    )
    parser.add_argument("--ir", type=int, default=5)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--venues", nargs="+", default=["CIKM", "KDD", "SIGIR"])
    parser.add_argument("--setting", choices=["test", "eval", "both"], default="both")
    parser.add_argument("--history-graph", default=None)
    parser.add_argument("--future-graph", default=None)
    parser.add_argument("--future-edge-totals", default=None)
    parser.add_argument("--final-csv", default=None)
    parser.add_argument("--cache-file", default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--citation-start-year", type=int, default=None)
    parser.add_argument("--citation-end-year", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = []
    if args.final_csv:
        if not (args.history_graph and args.cache_file and args.start_year and args.end_year):
            raise ValueError("--final-csv requires --history-graph, --cache-file, --start-year, and --end-year.")
        results.append(
            calculate_from_final_csv(
                Path(args.final_csv),
                Path(args.history_graph),
                Path(args.cache_file),
                args.start_year,
                args.end_year,
                args.citation_start_year or args.start_year,
                args.citation_end_year or args.end_year,
                args.ir,
            )
        )
    elif args.future_edge_totals:
        result = calculate_from_future_totals(Path(args.future_edge_totals), args.ir)
        results.append(result)
    elif args.history_graph or args.future_graph:
        if not (args.history_graph and args.future_graph and args.start_year and args.end_year):
            raise ValueError("--history-graph, --future-graph, --start-year, and --end-year are required together.")
        results.append(
            calculate_from_snapshots(
                Path(args.history_graph),
                Path(args.future_graph),
                args.start_year,
                args.end_year,
                args.ir,
            )
        )
    else:
        for job in default_snapshot_jobs(Path(args.dataset_root), args.venues, args.ir, args.setting):
            if not job["history_graph"].exists() or not job["future_graph"].exists():
                results.append(
                    {
                        "venue": job["venue"],
                        "setting": job["setting"],
                        "status": "missing_input",
                        "history_graph": str(job["history_graph"]),
                        "future_graph": str(job["future_graph"]),
                    }
                )
                continue
            result = calculate_from_snapshots(
                job["history_graph"],
                job["future_graph"],
                job["start_year"],
                job["end_year"],
                job["ir"],
            )
            result.update({"venue": job["venue"], "setting": job["setting"]})
            results.append(result)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
