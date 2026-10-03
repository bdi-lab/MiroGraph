"""
Create link prediction train/eval datasets from historical candidate pairs.

Historical candidate pairs are non-edges among concepts present in the
historical concept graph. Future papers provide labels by citation totals in a
future citation window. Train and test splits are balanced by default:
positive:negative = 1:1.
"""

import argparse
import csv
import logging
import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from openalex_utils import (
    OpenAlexAPIError,
    OpenAlexQuotaExceeded,
    OpenAlexClient,
    cache_record_to_counts,
    load_citation_cache,
    save_citation_cache,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Edge = Tuple[str, str]


def parse_concepts(value) -> list:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [c.strip() for c in str(value).split(",") if c.strip()]


def sorted_edge(c1: str, c2: str) -> Edge:
    return tuple(sorted((str(c1).strip(), str(c2).strip())))


def load_graph_edges_and_concepts(concept_graph_path: str) -> Tuple[Set[Edge], list]:
    graph_df = pd.read_csv(concept_graph_path)
    existing_edges = set()
    concepts = set()
    for _, row in graph_df.iterrows():
        edge = sorted_edge(row["concept1"], row["concept2"])
        existing_edges.add(edge)
        concepts.update(edge)
    return existing_edges, sorted(concepts)


def iter_candidate_pairs(concepts: list, existing_edges: Set[Edge]) -> Iterable[Edge]:
    for c1, c2 in combinations(concepts, 2):
        edge = (c1, c2)
        if edge not in existing_edges:
            yield edge


def collect_future_edges(
    final_csv_path: str,
    history_concepts: Set[str],
    history_edges: Set[Edge],
    future_years: Tuple[int, int],
) -> Tuple[Dict[Edge, list], Dict[int, dict]]:
    df = pd.read_csv(final_csv_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[
        df["year"].notna()
        & (df["year"] >= future_years[0])
        & (df["year"] <= future_years[1])
    ].copy()

    edge_to_papers = defaultdict(list)
    paper_data = {}

    for idx, row in df.iterrows():
        concepts = [c for c in parse_concepts(row.get("assigned_concepts", "")) if c in history_concepts]
        if len(concepts) < 2:
            continue
        paper_data[idx] = {
            "title": row.get("title", ""),
            "year": int(row["year"]),
        }
        for c1, c2 in combinations(sorted(set(concepts)), 2):
            edge = sorted_edge(c1, c2)
            if edge not in history_edges:
                edge_to_papers[edge].append(idx)

    logger.info(f"Future papers with candidate concepts: {len(paper_data):,}")
    logger.info(f"Future candidate edges: {len(edge_to_papers):,}")
    return edge_to_papers, paper_data


def fetch_paper_citations(
    paper_data: Dict[int, dict],
    citation_years: Tuple[int, int],
    cache_file: str,
    mailto: Optional[str] = None,
) -> Dict[int, Dict[int, int]]:
    cache = load_citation_cache(cache_file)
    out = {}
    hits = 0
    api_errors = 0
    new_records = 0
    max_workers = max(1, int(os.getenv("OPENALEX_MAX_WORKERS", "6")))
    logger.info(f"OpenAlex worker threads: {max_workers}")

    uncached_paper_ids = []
    for paper_id, item in paper_data.items():
        key = str(paper_id)
        counts = cache_record_to_counts(
            cache.get(key),
            item["year"],
            citation_years[0],
            citation_years[1],
        )
        if counts is not None:
            out[paper_id] = counts
            hits += 1
            continue
        uncached_paper_ids.append(paper_id)

    thread_state = threading.local()

    def get_client():
        if not hasattr(thread_state, "client"):
            thread_state.client = OpenAlexClient(mailto=mailto)
        return thread_state.client

    def fetch_one(paper_id):
        item = paper_data[paper_id]
        return paper_id, get_client().lookup_citations_by_title(item["title"], item["year"])

    if uncached_paper_ids:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_one, paper_id): paper_id
                for paper_id in uncached_paper_ids
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching future citations",
                unit="paper",
            ):
                paper_id = futures[future]
                item = paper_data[paper_id]
                key = str(paper_id)
                try:
                    _, record = future.result()
                except OpenAlexQuotaExceeded:
                    save_citation_cache(cache, cache_file)
                    raise
                except OpenAlexAPIError as exc:
                    logger.warning(f"OpenAlex API error for paper {paper_id} '{item['title'][:80]}...': {exc}")
                    out[paper_id] = {}
                    api_errors += 1
                    continue

                cache[key] = record
                out[paper_id] = cache_record_to_counts(
                    record,
                    item["year"],
                    citation_years[0],
                    citation_years[1],
                ) or {}
                new_records += 1

                if new_records % 50 == 0:
                    save_citation_cache(cache, cache_file)

    save_citation_cache(cache, cache_file)
    logger.info(f"Citation cache hits: {hits:,} / {len(paper_data):,}")
    logger.info(f"OpenAlex API errors not cached: {api_errors:,}")
    logger.info(f"New citation records cached: {new_records:,}")
    return out


def aggregate_edge_totals(edge_to_papers: Dict[Edge, list], paper_citations: Dict[int, Dict[int, int]]) -> Dict[Edge, int]:
    totals = {}
    for edge, paper_ids in tqdm(edge_to_papers.items(), desc="Aggregating future edge totals", unit="edge"):
        total = 0
        for paper_id in paper_ids:
            total += sum(int(v) for v in paper_citations.get(paper_id, {}).values())
        totals[edge] = total
    return totals


def sample_negative_edges(
    concepts: list,
    history_edges: Set[Edge],
    positive_edges: Set[Edge],
    exclude_edges: Set[Edge],
    n: int,
    seed: int,
    candidate_concepts: Optional[Set[str]] = None,
    exclude_within_concepts: Optional[Set[str]] = None,
    desc: str = "Sampling negatives",
) -> Set[Edge]:
    if n <= 0:
        return set()
    rng = random.Random(seed)
    sampling_concepts = sorted(candidate_concepts) if candidate_concepts is not None else concepts
    if len(sampling_concepts) < 2:
        return set()

    exclude_within_concepts = set(exclude_within_concepts or ())
    sample = set()

    def is_blocked(edge: Edge) -> bool:
        if edge in history_edges or edge in positive_edges or edge in exclude_edges or edge in sample:
            return True
        if exclude_within_concepts and edge[0] in exclude_within_concepts and edge[1] in exclude_within_concepts:
            return True
        return False

    max_attempts = max(10_000, n * 200)
    attempts = 0
    progress = tqdm(total=n, desc=desc, unit="edge", leave=False)
    try:
        while len(sample) < n and attempts < max_attempts:
            edge = sorted_edge(*rng.sample(sampling_concepts, 2))
            attempts += 1
            if is_blocked(edge):
                continue
            sample.add(edge)
            progress.update(1)
    finally:
        progress.close()

    if len(sample) < n:
        logger.info(f"{desc}: random sampler filled {len(sample):,}/{n:,}; scanning remaining pool")
        for edge in tqdm(iter_candidate_pairs(sampling_concepts, history_edges), desc=f"{desc} fallback", unit="pair"):
            if len(sample) >= n:
                break
            if is_blocked(edge):
                continue
            sample.add(edge)

    if len(sample) < n:
        logger.warning(f"Requested {n:,} negatives but only sampled {len(sample):,}")
    return sample


def sample_mixed_negative_edges(
    concepts: list,
    history_edges: Set[Edge],
    positive_edges: Set[Edge],
    edge_totals: Dict[Edge, int],
    threshold: int,
    exclude_edges: Set[Edge],
    n: int,
    seed: int,
    positive_concepts: Set[str],
    desc_prefix: str,
) -> Set[Edge]:
    """Sample half negatives from positive concepts and half from the remaining global pool.

    Within each source quota, prefer hard negatives with 0 < citations < threshold.
    """
    if n <= 0:
        return set()

    positive_concept_target = n // 2
    global_target = n - positive_concept_target

    positive_concept_hard_pool = {
        edge
        for edge, total in edge_totals.items()
        if 0 < int(total) < int(threshold)
        and edge[0] in positive_concepts
        and edge[1] in positive_concepts
        and edge not in history_edges
        and edge not in positive_edges
        and edge not in exclude_edges
    }
    positive_concept_hard = sample_edges_from_pool(
        positive_concept_hard_pool,
        positive_concept_target,
        seed=seed + 10_000,
    )
    positive_concept_negatives = set(positive_concept_hard)
    positive_concept_fill_needed = positive_concept_target - len(positive_concept_negatives)
    positive_concept_fill = sample_negative_edges(
        concepts,
        history_edges,
        positive_edges,
        exclude_edges=exclude_edges | positive_concept_negatives,
        n=positive_concept_fill_needed,
        seed=seed + 20_000,
        candidate_concepts=positive_concepts,
        desc=f"{desc_prefix} positive-concept negatives",
    )
    positive_concept_negatives.update(positive_concept_fill)

    selected = set(positive_concept_negatives)
    global_needed = global_target + (positive_concept_target - len(positive_concept_negatives))
    global_negatives = sample_negative_edges(
        concepts,
        history_edges,
        positive_edges,
        exclude_edges=exclude_edges | selected,
        n=global_needed,
        seed=seed + 70_000,
        exclude_within_concepts=positive_concepts,
        desc=f"{desc_prefix} global negatives",
    )
    selected.update(global_negatives)

    if len(selected) < n:
        fill_needed = n - len(selected)
        logger.info(f"{desc_prefix}: global pool short by {fill_needed:,}; filling from positive-concept pool")
        fill_negatives = sample_negative_edges(
            concepts,
            history_edges,
            positive_edges,
            exclude_edges=exclude_edges | selected,
            n=fill_needed,
            seed=seed + 100_000,
            candidate_concepts=positive_concepts,
            desc=f"{desc_prefix} fallback positives",
        )
        selected.update(fill_negatives)

    logger.info(
        f"{desc_prefix}: mixed negatives selected={len(selected):,}, "
        f"positive_concept_source={len(positive_concept_negatives):,}, "
        f"positive_concept_nonzero={len(positive_concept_hard):,}, "
        f"global_source={len(global_negatives):,}"
    )
    return selected


def sample_edges_from_pool(edges: Iterable[Edge], n: int, seed: int) -> Set[Edge]:
    """Uniformly sample up to n edges from a finite iterable."""
    if n <= 0:
        return set()
    rng = random.Random(seed)
    pool = sorted(set(edges))
    if len(pool) <= n:
        return set(pool)
    return set(rng.sample(pool, n))


def load_edge_totals(path: str) -> Dict[Edge, int]:
    totals = {}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        totals[sorted_edge(row["concept1"], row["concept2"])] = int(row["total_citations"])
    return totals


def write_rows(path: Path, rows: Iterable[Tuple[str, str, int, int]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept1", "concept2", "total_citations", "label"])
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def create_datasets(
    final_csv_path: str,
    concept_graph_path: str,
    output_dir: str,
    future_years: Tuple[int, int],
    citation_years: Tuple[int, int],
    thresholds: list,
    cache_file: str,
    train_ratio: float = 0.85,
    random_seed: int = 42,
    eval_remaining: bool = False,
    train_only: bool = False,
    prefix: str = "data",
    mailto: Optional[str] = None,
    use_openalex: bool = True,
    balanced_eval: bool = True,
    future_edge_totals_path: Optional[str] = None,
):
    history_edges, concepts = load_graph_edges_and_concepts(concept_graph_path)
    concept_set = set(concepts)
    total_candidates = len(concepts) * (len(concepts) - 1) // 2 - len(history_edges)
    logger.info(f"Historical concepts: {len(concepts):,}")
    logger.info(f"Historical edges: {len(history_edges):,}")
    logger.info(f"Historical candidate pairs: {total_candidates:,}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    positives_path = output / f"{prefix}_future_edge_totals_{future_years[0]}_{future_years[1]}.csv"

    if future_edge_totals_path:
        edge_totals = load_edge_totals(future_edge_totals_path)
        logger.info(f"Loaded future edge totals from: {future_edge_totals_path}")
    else:
        future_edge_to_papers, paper_data = collect_future_edges(
            final_csv_path,
            concept_set,
            history_edges,
            future_years,
        )
        if use_openalex:
            paper_citations = fetch_paper_citations(paper_data, citation_years, cache_file, mailto=mailto)
        else:
            logger.warning("OpenAlex disabled: all future citation totals will be 0")
            paper_citations = {paper_id: {} for paper_id in paper_data}
        edge_totals = aggregate_edge_totals(future_edge_to_papers, paper_citations)

        with open(positives_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["concept1", "concept2", "total_citations"])
            for (c1, c2), total in sorted(edge_totals.items(), key=lambda x: (-x[1], x[0])):
                writer.writerow([c1, c2, int(total)])
        logger.info(f"Future edge totals saved: {positives_path}")

    rng = random.Random(random_seed)
    for threshold in thresholds:
        positive_edges = {edge for edge, total in edge_totals.items() if int(total) >= int(threshold)}
        positives = sorted(positive_edges)
        rng.shuffle(positives)
        positive_concepts = {concept for edge in positive_edges for concept in edge}
        positive_concept_candidates = (
            len(positive_concepts) * (len(positive_concepts) - 1) // 2
            - sum(1 for edge in history_edges if edge[0] in positive_concepts and edge[1] in positive_concepts)
        )
        positive_concept_negatives = max(0, positive_concept_candidates - len(positive_edges))
        logger.info(
            f"IR={threshold}: negative pool restricted to positive concepts; "
            f"positive_concepts={len(positive_concepts):,}, "
            f"candidate_pairs={positive_concept_candidates:,}, "
            f"label0_pool={positive_concept_negatives:,}"
        )

        split = int(round(len(positives) * train_ratio))
        train_pos = positives[:split]
        heldout_pos = [] if train_only else positives[split:]

        train_pos_set = set(train_pos)
        n_negative_needed = len(train_pos)
        if eval_remaining and balanced_eval and not train_only:
            n_negative_needed += len(heldout_pos)

        sampled_negatives = sample_mixed_negative_edges(
            concepts,
            history_edges,
            positive_edges,
            edge_totals,
            int(threshold),
            exclude_edges=set(),
            n=n_negative_needed,
            seed=random_seed + int(threshold),
            positive_concepts=positive_concepts,
            desc_prefix=f"IR={threshold}",
        )

        inside_negatives = [
            edge for edge in sampled_negatives
            if edge[0] in positive_concepts and edge[1] in positive_concepts
        ]
        global_negatives = [
            edge for edge in sampled_negatives
            if not (edge[0] in positive_concepts and edge[1] in positive_concepts)
        ]
        rng.shuffle(inside_negatives)
        rng.shuffle(global_negatives)

        train_inside_target = len(train_pos) // 2
        train_global_target = len(train_pos) - train_inside_target
        train_neg_set = set(inside_negatives[:train_inside_target]) | set(global_negatives[:train_global_target])

        eval_neg_set = set()
        if eval_remaining and balanced_eval and not train_only:
            eval_inside_target = len(heldout_pos) // 2
            eval_global_target = len(heldout_pos) - eval_inside_target
            eval_neg_set = (
                set(inside_negatives[train_inside_target:train_inside_target + eval_inside_target])
                | set(global_negatives[train_global_target:train_global_target + eval_global_target])
            )

        if len(train_neg_set) < len(train_pos):
            fill_needed = len(train_pos) - len(train_neg_set)
            logger.warning(f"IR={threshold}: train negatives short by {fill_needed:,}; filling from sampled remainder")
            remainder = list(sampled_negatives - train_neg_set - eval_neg_set)
            rng.shuffle(remainder)
            train_neg_set.update(remainder[:fill_needed])

        if eval_remaining and balanced_eval and not train_only and len(eval_neg_set) < len(heldout_pos):
            fill_needed = len(heldout_pos) - len(eval_neg_set)
            logger.warning(f"IR={threshold}: eval negatives short by {fill_needed:,}; filling from sampled remainder")
            remainder = list(sampled_negatives - train_neg_set - eval_neg_set)
            rng.shuffle(remainder)
            eval_neg_set.update(remainder[:fill_needed])

        train_edges = train_pos_set | train_neg_set

        train_path = output / f"{prefix}_train_ir{threshold}.csv"
        train_rows = []
        for edge in train_pos:
            train_rows.append((edge[0], edge[1], int(edge_totals.get(edge, 0)), 1))
        for edge in sorted(train_neg_set):
            train_rows.append((edge[0], edge[1], int(edge_totals.get(edge, 0)), 0))
        rng.shuffle(train_rows)
        n_train = write_rows(train_path, train_rows)

        logger.info(
            f"IR={threshold}: positives={len(positive_edges):,}, "
            f"train_pos={len(train_pos):,}, train_neg={len(train_neg_set):,}, train_rows={n_train:,}"
        )

        if train_only:
            continue

        if eval_remaining:
            eval_path = output / f"{prefix}_test_ir{threshold}.csv"
            if balanced_eval:
                eval_rows = []
                for edge in heldout_pos:
                    eval_rows.append((edge[0], edge[1], int(edge_totals.get(edge, 0)), 1))
                for edge in sorted(eval_neg_set):
                    eval_rows.append((edge[0], edge[1], int(edge_totals.get(edge, 0)), 0))
                rng.shuffle(eval_rows)
                n_eval = write_rows(eval_path, eval_rows)
                logger.info(
                    f"IR={threshold}: balanced eval rows written={n_eval:,}, "
                    f"eval_pos={len(heldout_pos):,}, eval_neg={len(eval_neg_set):,}"
                )
                continue

            def eval_rows():
                for edge in tqdm(iter_candidate_pairs(concepts, history_edges), desc=f"Writing eval IR={threshold}", unit="pair"):
                    if edge in train_edges:
                        continue
                    total = int(edge_totals.get(edge, 0))
                    yield (edge[0], edge[1], total, int(total >= int(threshold)))

            n_eval = write_rows(eval_path, eval_rows())
            logger.info(f"IR={threshold}: eval rows written={n_eval:,}, heldout positives={len(heldout_pos):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create link prediction train/eval datasets")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--concept-graph", "-g", required=True)
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--future-years", nargs=2, type=int, required=True)
    parser.add_argument("--citation-years", nargs=2, type=int, required=True)
    parser.add_argument("--thresholds", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--eval-remaining", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--prefix", default="data")
    parser.add_argument("--mailto", default=None)
    parser.add_argument("--no-openalex", action="store_true")
    parser.add_argument(
        "--eval-all-remaining",
        action="store_true",
        help="Write every non-train candidate pair to eval/test instead of a balanced 1:1 split.",
    )
    parser.add_argument(
        "--future-edge-totals",
        default=None,
        help="Use an existing future edge totals CSV and only regenerate train/test splits.",
    )
    args = parser.parse_args()

    create_datasets(
        final_csv_path=args.input,
        concept_graph_path=args.concept_graph,
        output_dir=args.output_dir,
        future_years=tuple(args.future_years),
        citation_years=tuple(args.citation_years),
        thresholds=args.thresholds,
        cache_file=args.cache,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        eval_remaining=args.eval_remaining,
        train_only=args.train_only,
        prefix=args.prefix,
        mailto=args.mailto,
        use_openalex=not args.no_openalex,
        balanced_eval=not args.eval_all_remaining,
        future_edge_totals_path=args.future_edge_totals,
    )
