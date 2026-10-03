"""
Data Loading and Preprocessing Module
Loads and preprocesses concept keywords per paper from CSV files.
"""

import pandas as pd
import logging
import ast
import re
from typing import List, Dict, Tuple, Union
from collections import Counter


class ConceptDataLoader:
    """Concept keyword data loader and preprocessor"""
    
    def __init__(self, min_frequency: int = 1, normalize_concepts: bool = True):
        """
        Args:
            min_frequency: Minimum raw keyword frequency threshold before clustering.
                Keep this low so rare surface forms can still join semantic clusters.
            normalize_concepts: Whether to apply lightweight lexical normalization.
        """
        self.min_frequency = min_frequency
        self.normalize_concepts = normalize_concepts
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def normalize_concept_columns(concept_column: Union[str, List[str], Tuple[str, ...]]) -> List[str]:
        """Normalize a concept-column argument into a list of column names."""
        if isinstance(concept_column, (list, tuple)):
            return [str(c).strip() for c in concept_column if str(c).strip()]
        if isinstance(concept_column, str):
            return [c.strip() for c in concept_column.split(',') if c.strip()]
        return [str(concept_column).strip()]

    @staticmethod
    def normalize_concept_text(concept: str) -> str:
        """Apply lightweight lexical normalization without semantic rewriting."""
        value = str(concept).strip().lower()
        value = value.replace('_', ' ')
        value = re.sub(r'[-/]+', ' ', value)
        value = re.sub(r'\s+', ' ', value).strip()

        # Remove short parenthetical acronyms/descriptors: "graph neural networks (gnns)".
        value = re.sub(r'\s*\(([a-z0-9 ]{1,16})\)\s*$', '', value).strip()
        value = re.sub(r'\s+', ' ', value).strip()
        return value

    @classmethod
    def _clean_concept(cls, concept: str, normalize: bool = True) -> str:
        value = str(concept).strip()
        if not value:
            return ''
        return cls.normalize_concept_text(value) if normalize else value

    @classmethod
    def parse_concepts_value(cls, concepts, normalize: bool = True) -> List[str]:
        """Parse one concept cell, supporting Python-list strings and comma-separated strings."""
        if isinstance(concepts, list):
            return [cleaned for c in concepts if (cleaned := cls._clean_concept(c, normalize))]

        if concepts is None or pd.isna(concepts):
            return []

        if not isinstance(concepts, str):
            value = cls._clean_concept(concepts, normalize)
            return [value] if value else []

        value = concepts.strip()
        if not value:
            return []

        if value.startswith('[') and value.endswith(']'):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [cleaned for c in parsed if (cleaned := cls._clean_concept(c, normalize))]
            except (ValueError, SyntaxError):
                pass

        return [
            cleaned
            for c in value.split(',')
            if (cleaned := cls._clean_concept(c.strip().strip("'\""), normalize))
        ]

    @classmethod
    def parse_row_concepts(cls, row: pd.Series,
                           concept_column: Union[str, List[str], Tuple[str, ...]],
                           normalize: bool = True) -> List[str]:
        """Parse and de-duplicate concepts from one or more concept columns in a row."""
        concepts = []
        seen = set()
        for column in cls.normalize_concept_columns(concept_column):
            for concept in cls.parse_concepts_value(row.get(column, None), normalize=normalize):
                if concept not in seen:
                    concepts.append(concept)
                    seen.add(concept)
        return concepts
        
    def load_from_csv(self, csv_path: str,
                      concept_column: Union[str, List[str], Tuple[str, ...]] = 'concepts') -> pd.DataFrame:
        """
        Load concept keywords per paper from CSV file
        
        Args:
            csv_path: CSV file path
            concept_column: Column name containing concept keywords
            
        Returns:
            DataFrame with paper_id and concepts
        """
        self.logger.info(f"Loading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        missing_columns = [
            column for column in self.normalize_concept_columns(concept_column)
            if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Concept column(s) not found in DataFrame: {missing_columns}")
        
        self.logger.info(f"Loaded papers: {len(df)}")
        return df
    
    def extract_all_concepts(self, df: pd.DataFrame,
                             concept_column: Union[str, List[str], Tuple[str, ...]] = 'concepts') -> List[str]:
        """
        Extract and deduplicate concept keywords from all papers
        
        Args:
            df: Paper DataFrame
            concept_column: Concept keyword column name
            
        Returns:
            List of all unique concept keywords
        """
        self.logger.info("Extracting all concept keywords...")
        
        all_concepts = []
        for idx, row in df.iterrows():
            all_concepts.extend(
                self.parse_row_concepts(row, concept_column, normalize=self.normalize_concepts)
            )
        
        # Remove duplicates
        unique_concepts = list(set(all_concepts))
        
        self.logger.info(f"Total concept keywords: {len(all_concepts)}")
        self.logger.info(f"Unique concept keywords: {len(unique_concepts)}")
        
        return unique_concepts
    
    def filter_by_frequency(self, df: pd.DataFrame,
                           concept_column: Union[str, List[str], Tuple[str, ...]] = 'concepts',
                           unique_concepts: List[str] = None) -> Tuple[List[str], Dict[str, int]]:
        """
        Filter keywords that appear at least min_frequency times
        
        Args:
            df: Paper DataFrame
            concept_column: Concept keyword column name
            unique_concepts: Unique concept list (auto-extracted if None)
            
        Returns:
            (Filtered concept list, concept frequency dictionary)
        """
        if unique_concepts is None:
            unique_concepts = self.extract_all_concepts(df, concept_column)
        
        self.logger.info(f"Calculating concept keyword frequencies... (min frequency: {self.min_frequency})")
        
        # Calculate frequencies
        concept_counter = Counter()
        for idx, row in df.iterrows():
            concept_counter.update(
                self.parse_row_concepts(row, concept_column, normalize=self.normalize_concepts)
            )
        
        # Filter by frequency
        filtered_concepts = [
            concept for concept in unique_concepts 
            if concept_counter[concept] >= self.min_frequency
        ]
        
        filtered_out = len(unique_concepts) - len(filtered_concepts)
        
        self.logger.info(f"Before filtering: {len(unique_concepts)}")
        self.logger.info(f"After filtering: {len(filtered_concepts)} (removed: {filtered_out})")
        
        return filtered_concepts, dict(concept_counter)
    
    def create_concept_to_papers_mapping(self, df: pd.DataFrame, 
                                        concept_column: Union[str, List[str], Tuple[str, ...]] = 'concepts',
                                        paper_id_column: str = 'paper_id') -> Dict[str, List[str]]:
        """
        Create mapping of which papers each concept appears in
        
        Args:
            df: Paper DataFrame
            concept_column: Concept keyword column name
            paper_id_column: Paper ID column name
            
        Returns:
            {concept: [paper_id1, paper_id2, ...]} dictionary
        """
        self.logger.info("Creating concept-paper mapping...")
        
        concept_to_papers = {}
        
        for idx, row in df.iterrows():
            paper_id = row[paper_id_column] if paper_id_column in df.columns else idx
            concept_list = self.parse_row_concepts(
                row, concept_column, normalize=self.normalize_concepts
            )
            
            for concept in concept_list:
                if concept not in concept_to_papers:
                    concept_to_papers[concept] = []
                concept_to_papers[concept].append(str(paper_id))
        
        self.logger.info(f"Mapping created: {len(concept_to_papers)} concepts")
        
        return concept_to_papers
