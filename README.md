# Bridging Concepts: Multi-Agent Reasoning over Scientific Concept Graphs for High-Impact Research Prediction

This is the official implementation of the paper 

**Bridging Concepts: Multi-Agent Reasoning over Scientific Concept Graphs for High-Impact Research Prediction**

The 3rd Workshop on Knowledge-Enhanced Information Retrieval at Conference on Information and Knowledge Management (KEIR @ CIKM 2026).

All codes are written by **Minsung Hwang** ([hminsung@kaist.ac.kr](mailto:hminsung@kaist.ac.kr)) and **Donggyu Yoon** ([justinyoonwk@kaist.ac.kr](mailto:justinyoonwk@kaist.ac.kr)).

If you use this repository, please cite our paper.

```bibtex
@article{mirograph,
  author={Donggyu Yoon and Minsung Hwang and Juneha Baek and Donghyuk Shin and Joyce Jiyoung Whang},
  title={Bridging Concepts: Multi-Agent Reasoning over Scientific Concept Graphs for High-Impact Research Prediction},
  year={2026},
  journal={arXiv preprint arXiv.2610.25281},
  doi = {10.48550/arXiv.2610.25281}
}
```

## Requirements

- Python 3.11
- `pip install -r requirements.txt`
- Set `OPENAI_API_KEY` in the environment.

## Repository Structure

- `concept_extraction_code/`: extracts hierarchical scientific concepts from venue BibTeX files.
- `scg_dataset_pipeline_code/`: clusters extracted concepts, builds temporal Scientific Concept Graphs, and creates test/eval prediction datasets.
- `mirograph_model_code/`: runs MiroGraph multi-agent scoring, classifier design, baselines, and ablations.
- `dataset/`: public CIKM, KDD, and SIGIR graph/dataset files.

## Concept Extraction

Run from `concept_extraction_code/`. The extraction script supports CIKM, KDD, and SIGIR through 2025.

```bash
cd concept_extraction_code
python run_conference.py --conference CIKM --output-dir output
```

Outputs are written as `output/<venue>_<start>_2025_concepts.csv` and `.pkl`.

## SCG Dataset Pipeline

Run from `scg_dataset_pipeline_code/`. The retained files are the clustering/naming pipeline, graph builder, OpenAlex citation helper, and link-prediction dataset generator.

```bash
cd scg_dataset_pipeline_code
python main.py \
  --input ../concept_extraction_code/output/cikm_1993_2025_concepts.csv \
  --venue CIKM \
  --output-dir output/cikm

python build_concept_graph.py \
  --input output/cikm/final_papers_with_concepts.csv \
  --venue CIKM \
  --output-dir ../dataset/CIKM/test \
  --max-year 2023 \
  --cache ../dataset/CIKM/openalex_paper_cache.json \
  --output-filename concept_graph_2023.csv

python create_link_prediction_datasets.py \
  --input output/cikm/final_papers_with_concepts.csv \
  --concept-graph ../dataset/CIKM/test/concept_graph_2023.csv \
  --output-dir ../dataset/CIKM/test \
  --future-years 2024 2025 \
  --citation-years 2024 2025 \
  --thresholds 1 3 5 10 \
  --cache ../dataset/CIKM/openalex_paper_cache.json \
  --eval-remaining \
  --prefix test
```

### Dataset Labels

For each historical graph snapshot, candidate pairs are concept pairs that are not connected in the historical SCG. A positive pair is an unseen pair whose accumulated citations in the future window are at least the impact threshold `IR`. All remaining unseen candidate pairs form the negative pool, and the dataset generator samples the same number of negative pairs as positive pairs to balance each split.

For the public `IR=5` splits, the positive-class base rate is 50.0%. Across CIKM, KDD, and SIGIR, positive pairs have future-window citation totals with median 9, IQR 6-14, mean 13.09, and range 5-247. Among negative pairs, 63.9% have zero future citations and 36.1% are hard negatives with nonzero citations below `IR`.

Venue-specific `IR=5` citation distributions are:

| Venue | Rows | Positive Citation Median | IQR | Mean | Range | Negative Zero / Hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIKM | 3,864 | 8 | 6-11 | 9.98 | 5-43 | 64.8% / 35.2% |
| KDD | 5,206 | 9 | 6-14 | 12.94 | 5-181 | 62.4% / 37.6% |
| SIGIR | 4,942 | 10 | 7-17 | 15.69 | 5-247 | 64.9% / 35.1% |

## Model Defaults

Defaults are defined in `mirograph_model_code/mirograph_config.py`.

- Agent scoring / direct judging: `gpt-4o`
- Feature schema design and structural feature code generation: `gpt-5.4`
- Classifier code generation: `gpt-5.4`

These defaults can be overridden with CLI flags or environment variables:

- `MIROGRAPH_SCORING_MODEL`
- `MIROGRAPH_CODE_MODEL`

## Feature Design

Run this once per dataset/setting before multi-agent scoring.

```bash
python mirograph_model_code/feature_designer.py \
  --dataset CIKM \
  --setting test \
  --ir 5 \
  --output_schema results/feature_designs/cikm_test_ir5_schema.json \
  --output_structure_code results/feature_designs/cikm_test_ir5_structure.py
```

## Test

Test uses the `test` split. Both train and test scoring contexts use the 2023 graph.

Expected files:

- `dataset/<DATASET>/test/concept_graph_2023.csv`
- `dataset/<DATASET>/test/test_train_ir<IR>.csv`
- `dataset/<DATASET>/test/test_test_ir<IR>.csv`

```bash
python mirograph_model_code/multiagent_scoring.py \
  --experiment test \
  --dataset CIKM \
  --ir 5 \
  --feature_schema results/feature_designs/cikm_test_ir5_schema.json \
  --structure_code results/feature_designs/cikm_test_ir5_structure.py \
  --output_dir results/cikm_test_ir5

python mirograph_model_code/classifier_designer.py \
  --train_scores_cache results/cikm_test_ir5/train_scores.jsonl \
  --test_scores_cache results/cikm_test_ir5/test_scores.jsonl \
  --feature_schema results/feature_designs/cikm_test_ir5_schema.json \
  --output results/cikm_test_ir5/meta_code_test_predictions.jsonl \
  --best_code_out results/cikm_test_ir5/best_code.py
```

## Eval

Eval trains on the `eval` train file and evaluates on the same held-out test file used by Test.
Train scoring uses the 2021 graph. Eval target scoring uses the 2023 test graph.

Expected files:

- `dataset/<DATASET>/eval/concept_graph_2021.csv`
- `dataset/<DATASET>/eval/eval_train_ir<IR>.csv`
- `dataset/<DATASET>/test/concept_graph_2023.csv`
- `dataset/<DATASET>/test/test_test_ir<IR>.csv`

```bash
python mirograph_model_code/multiagent_scoring.py \
  --experiment eval \
  --dataset CIKM \
  --ir 5 \
  --feature_schema results/feature_designs/cikm_test_ir5_schema.json \
  --structure_code results/feature_designs/cikm_test_ir5_structure.py \
  --output_dir results/cikm_eval_ir5

python mirograph_model_code/classifier_designer.py \
  --train_scores_cache results/cikm_eval_ir5/train_scores.jsonl \
  --test_scores_cache results/cikm_eval_ir5/test_scores.jsonl \
  --feature_schema results/feature_designs/cikm_test_ir5_schema.json \
  --output results/cikm_eval_ir5/meta_code_eval_predictions.jsonl \
  --best_code_out results/cikm_eval_ir5/best_code.py
```

Metrics are accuracy and macro-F1. The classifier designer splits the training score cache using a 70:15 train/validation weight ratio, then retrains the selected classifier on the full training cache before predicting test/eval samples.

## Ablations

Supported ablations are `no_structure`, `no_semantic`, `no_trend`, `single_score`, and the MLP classifier ablation.

```bash
python mirograph_model_code/ablation_study.py \
  --modes no_structure,no_semantic,no_trend \
  --train_scores_cache results/cikm_test_ir5/train_scores.jsonl \
  --test_scores_cache results/cikm_test_ir5/test_scores.jsonl \
  --feature_schema results/feature_designs/cikm_test_ir5_schema.json \
  --output results/cikm_test_ir5/ablation.jsonl
```

```bash
python mirograph_model_code/single_score_scoring.py \
  --experiment test \
  --dataset CIKM \
  --ir 5 \
  --output_dir results/cikm_test_ir5_single_score

python mirograph_model_code/ablation_study.py \
  --mode single_score \
  --train_scores_cache results/cikm_test_ir5_single_score/train_scores.jsonl \
  --test_scores_cache results/cikm_test_ir5_single_score/test_scores.jsonl \
  --output results/cikm_test_ir5_single_score/ablation_single_score.jsonl
```

```bash
python mirograph_model_code/ablation_study.py \
  --classifier_backend mlp \
  --train_scores_cache results/cikm_test_ir5/train_scores.jsonl \
  --test_scores_cache results/cikm_test_ir5/test_scores.jsonl \
  --feature_schema results/feature_designs/cikm_test_ir5_schema.json \
  --output results/cikm_test_ir5/ablation_mlp.jsonl
```

## Single-LLM Baseline

```bash
python mirograph_model_code/simple_prediction.py \
  --test_file dataset/CIKM/test/test_test_ir5.csv \
  --ir 5 \
  --model gpt-4o-mini \
  --output results/cikm_test_ir5/simple_gpt4o_mini.jsonl
```

## Licence

This code is released under the **CC BY-NC-SA 4.0** license.
