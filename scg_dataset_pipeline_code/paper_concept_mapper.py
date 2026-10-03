"""
Paper-Concept Mapping Module
Assigns final cluster representative concepts to each paper.
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple, Union

from data_loader import ConceptDataLoader


class PaperConceptMapper:
    """Assign final concepts to papers"""
    
    def __init__(self, normalize_concepts: bool = True):
        self.normalize_concepts = normalize_concepts
        self.logger = logging.getLogger(__name__)
    
    def map_papers_to_concepts(self,
                              df: pd.DataFrame,
                              concept_column: Union[str, List[str], Tuple[str, ...]],
                              concept_to_clusters: Dict[str, List[int]],
                              cluster_to_name: Dict[int, str],
                              paper_id_column: str = 'paper_id') -> pd.DataFrame:
        """
        Assign final concepts to each paper
        (keywords can belong to multiple clusters)
        
        Args:
            df: Paper DataFrame
            concept_column: Concept keyword column name
            concept_to_clusters: {concept: [cluster_id1, cluster_id2, ...]} mapping
            cluster_to_name: {cluster_id: concept_name} mapping
            paper_id_column: Paper ID column name
            
        Returns:
            DataFrame with assigned concepts
        """
        self.logger.info("Paper-concept mapping started...")
        
        # Collect concept keywords for each paper
        # Use index as key to handle duplicate paper_ids correctly
        paper_concepts = {}  # {index: concept_list}
        
        for idx, row in df.iterrows():
            paper_concepts[idx] = ConceptDataLoader.parse_row_concepts(
                row, concept_column, normalize=self.normalize_concepts
            )
        
        # Find clusters assigned to each paper (remove duplicates)
        # Use index as key to handle duplicate paper_ids correctly
        paper_to_clusters = {}  # {index: set of cluster_ids}
        
        for idx, concepts in paper_concepts.items():
            cluster_set = set()
            for concept in concepts:
                if concept in concept_to_clusters:
                    cluster_ids = concept_to_clusters[concept]
                    for cluster_id in cluster_ids:
                        if cluster_id in cluster_to_name:
                            cluster_set.add(cluster_id)
            paper_to_clusters[idx] = cluster_set
        
        # Assign concepts to DataFrame
        result_df = df.copy()
        
        # Create final concept list for each paper (remove duplicates)
        final_concepts = []
        for idx, row in result_df.iterrows():
            cluster_ids = paper_to_clusters.get(idx, set())
            concept_names = sorted(set([cluster_to_name[cid] for cid in cluster_ids if cid in cluster_to_name]))
            final_concepts.append(concept_names)
        
        result_df['assigned_concepts'] = final_concepts
        
        # Statistics
        total_papers = len(result_df)
        papers_with_concepts = sum(1 for concepts in final_concepts if len(concepts) > 0)
        total_concept_assignments = sum(len(concepts) for concepts in final_concepts)
        
        self.logger.info(f"Mapping completed:")
        self.logger.info(f"  - Total papers: {total_papers}")
        self.logger.info(f"  - Papers with assigned concepts: {papers_with_concepts} ({papers_with_concepts/total_papers*100:.2f}%)")
        self.logger.info(f"  - Total concept assignments: {total_concept_assignments}")
        self.logger.info(f"  - Average concepts per paper: {total_concept_assignments/total_papers:.2f}")
        
        return result_df
