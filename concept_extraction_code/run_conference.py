import os
import glob
import re
import logging
import argparse
from pathlib import Path
import pandas as pd
import bibtexparser

from src import AbstractStructurer, ConceptExtractor
from src.utils import Config, setup_logging, get_tqdm


TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed} | eta {remaining}]"
DEFAULT_BATCH_SIZE = 50

CONFERENCE_SETTINGS = {
    "sigir": {
        "folder": "SIGIR",
        "start_year": 2000,
        "end_year": 2025,
        "result_filename": "sigir_2000_2025_concepts",
    },
    "cikm": {
        "folder": "CIKM",
        "start_year": 1993,
        "end_year": 2025,
        "result_filename": "cikm_1993_2025_concepts",
    },
    "kdd": {
        "folder": "KDD",
        "start_year": 1995,
        "end_year": 2025,
        "result_filename": "kdd_1995_2025_concepts",
    },
}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[{}]", "", text)
    text = text.replace("\n", " ").strip()
    return text


def _coerce_year(value):
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_bibtex_files(conference_key, config):
    settings = CONFERENCE_SETTINGS[conference_key]
    directory_path = Path("conference") / settings["folder"]
    start_year = settings["start_year"]
    end_year = settings["end_year"]
    all_papers = []
    bib_files = sorted(glob.glob(str(directory_path / "*.bib")))
    print(f"Found {len(bib_files)} BibTeX files in {directory_path}")

    for bib_file in get_tqdm(
        config,
        bib_files,
        desc="Parsing BibTeX files",
        bar_format=TQDM_BAR_FORMAT,
    ):
        filename = os.path.basename(bib_file)
        year_match = re.search(r"(19|20)\\d{2}", filename)
        file_year = _coerce_year(year_match.group(0)) if year_match else None

        if file_year is not None and (file_year < start_year or file_year > end_year):
            continue

        with open(bib_file, "r", encoding="utf-8") as bibtex_file:
            parser = bibtexparser.bparser.BibTexParser(common_strings=True)
            try:
                bib_database = bibtexparser.load(bibtex_file, parser=parser)
                for entry in bib_database.entries:
                    if "abstract" not in entry:
                        continue
                    title = clean_text(entry.get("title", ""))
                    abstract = clean_text(entry.get("abstract", ""))
                    if not title or not abstract:
                        continue

                    entry_year = _coerce_year(entry.get("year")) or file_year
                    if entry_year is None or entry_year < start_year or entry_year > end_year:
                        continue

                    all_papers.append(
                        {
                            "title": title,
                            "abstract": abstract,
                            "year": entry_year,
                            "venue": settings["folder"],
                            "bib_id": entry.get("ID", ""),
                        }
                    )
            except Exception as e:
                print(f"Error parsing {filename}: {e}")

    df = pd.DataFrame(all_papers)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["bib_id", "title"]).reset_index(drop=True)
    return df


def process_in_batches(processor, df, step_name, config, batch_size=DEFAULT_BATCH_SIZE, save_path=None):
    logger = logging.getLogger(__name__)
    processed_df = pd.DataFrame()
    start_idx = 0

    if save_path and os.path.exists(save_path):
        try:
            processed_df = pd.read_pickle(save_path)
            if len(processed_df) > 0:
                start_idx = len(processed_df)
                logger.info(
                    f"Resuming {step_name} from index {start_idx} "
                    f"(loaded {len(processed_df)} processed papers)"
                )
        except Exception as e:
            logger.warning(f"Failed to load intermediate file: {e}. Starting from scratch.")

    if start_idx < len(df):
        total_batches = (len(df) - start_idx + batch_size - 1) // batch_size
        batch_range = range(start_idx, len(df), batch_size)
        for batch_index, i in enumerate(
            get_tqdm(
                config,
                batch_range,
                total=total_batches,
                desc=f"{step_name} batches",
                bar_format=TQDM_BAR_FORMAT,
            )
        ):
            end_idx = min(i + batch_size, len(df))
            batch = df.iloc[i:end_idx].copy()
            logger.info(
                f"Processing {step_name} batch {batch_index + 1} "
                f"(Indices {i} to {end_idx})"
            )

            try:
                if step_name == "Structuring":
                    result_batch = processor.structure_abstracts(batch)
                elif step_name == "Extraction":
                    result_batch = processor.extract_concepts(batch)
                else:
                    raise ValueError(f"Unknown step name: {step_name}")

                processed_df = pd.concat([processed_df, result_batch], ignore_index=True)

                if save_path:
                    processed_df.to_pickle(save_path)
            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                raise e

    return processed_df


def resolve_conference(args):
    if args.conference:
        return args.conference
    if args.sigir:
        return "sigir"
    if args.kdd:
        return "kdd"
    if args.cikm:
        return "cikm"
    raise ValueError("Conference selection missing")


def main():
    parser = argparse.ArgumentParser(description="Run concept extraction pipeline for a conference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--conference", type=str.lower, choices=["sigir", "kdd", "cikm"])
    group.add_argument("--sigir", action="store_true", help="Run SIGIR pipeline")
    group.add_argument("--kdd", action="store_true", help="Run KDD pipeline")
    group.add_argument("--cikm", action="store_true", help="Run CIKM pipeline")

    parser.add_argument("--config", "-c", default="config/config.yaml", help="Config file path")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")

    args = parser.parse_args()
    conference_key = resolve_conference(args)
    settings = CONFERENCE_SETTINGS[conference_key]

    config = Config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Loading {settings['folder']} BibTeX files ({settings['start_year']}-{settings['end_year']}) "
        f"from conference/{settings['folder']}..."
    )
    df = parse_bibtex_files(conference_key, config)
    logger.info(f"Total papers loaded: {len(df)}")

    if len(df) == 0:
        logger.error("No papers found. Check directory.")
        return

    logger.info("Starting Abstract Structuring...")
    structurer = AbstractStructurer(args.config)
    struct_save_path = output_dir / f"temp_{conference_key}_all_structured.pkl"
    df = process_in_batches(
        structurer,
        df,
        "Structuring",
        config,
        batch_size=args.batch_size,
        save_path=struct_save_path,
    )

    df = structurer.get_valid_papers(df)
    logger.info(f"Valid papers after structuring: {len(df)}")

    logger.info("Starting Hierarchical Concept Extraction...")
    extractor = ConceptExtractor(args.config)
    extract_save_path = output_dir / f"temp_{conference_key}_all_extracted.pkl"
    df = process_in_batches(
        extractor,
        df,
        "Extraction",
        config,
        batch_size=args.batch_size,
        save_path=extract_save_path,
    )

    target_columns = [
        "year",
        "title",
        "level_0_concepts",
        "level_1_concepts",
        "level_2_concepts",
        "level_3_concepts",
        "level_4_concepts",
        "level_5_concepts",
    ]
    existing_cols = [col for col in target_columns if col in df.columns]
    final_df = df[existing_cols].copy()

    final_df["year"] = pd.to_numeric(final_df["year"], errors="coerce")
    final_df = final_df.sort_values(["year", "title"]).reset_index(drop=True)

    result_filename = settings["result_filename"]
    output_pkl = output_dir / f"{result_filename}.pkl"
    output_csv = output_dir / f"{result_filename}.csv"
    final_df.to_pickle(output_pkl)
    final_df.to_csv(output_csv, index=False)

    logger.info(f"All processing complete. Saved {len(final_df)} papers.")
    logger.info(f"File saved to: {output_pkl}")


if __name__ == "__main__":
    main()
