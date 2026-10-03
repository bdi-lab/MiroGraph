"""
Concept Extractor Module
Extracts hierarchical concepts from structured abstracts using LLM
"""

import json
import logging
import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path

from .utils import Config, LLMClient, PromptLoader, setup_logging, validate_json_response, save_intermediate_result


class ConceptExtractor:
    """Class for extracting concepts from structured abstracts"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.client = LLMClient(self.config)
        self.prompt_loader = PromptLoader(self.config)
        
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
                                        
        self.prompt = self.prompt_loader.load_prompt(
            self.config.get('prompts.concept_extractor', 'hierarchical_concept_extraction.txt')
        )
    
    def extract_concepts(self, df: pd.DataFrame, 
                        structured_abs_col: str = 'structured_abs') -> pd.DataFrame:
        """
        Extract hierarchical concepts from structured abstracts
        
        Args:
            df: DataFrame with structured abstracts
            structured_abs_col: Column name containing structured abstracts
            
        Returns:
            DataFrame with extracted concepts
        """
        self.logger.info(f"Starting concept extraction for {len(df)} papers")
        
                                    
        concept_responses = self.client.process_batch_parallel(
            self.prompt, df[structured_abs_col].tolist()
        )
        
        df['c_list'] = concept_responses
        
                                                    
        concept_data = self._parse_concept_responses(df['c_list'])
        
                                              
        for col, data in concept_data.items():
            df[col] = data
        
                                                 
        if self.config.get('processing.save_intermediate_results', True):
            output_dir = Path(self.config.get('paths.output_dir', './output'))
            output_dir.mkdir(parents=True, exist_ok=True)
            save_intermediate_result(
                df, 
                output_dir / 'extracted_concepts.pkl',
                'extracted concepts'
            )
        
        self.logger.info("Concept extraction completed")
        return df
    
    def _parse_concept_responses(self, concept_responses: pd.Series) -> Dict[str, List]:
        """Parse concept extraction responses into structured format"""
        
        def extract_concepts_and_methods(hierarchical_json: Dict) -> Dict[str, List]:
            """Extract concepts by level and methods from hierarchical JSON"""
            concepts_by_level = {}
            for i in range(0, 6):                      
                level_key = f'level_{i}'
                if (hierarchical_json.get('hierarchical_concepts', {}).get(level_key)):
                    concepts_by_level[level_key] = [
                        item['term'].lower() 
                        for item in hierarchical_json['hierarchical_concepts'][level_key]
                    ]
                else:
                    concepts_by_level[level_key] = []
            
            methods = hierarchical_json.get('methods', [])
            return concepts_by_level, methods
        
        def apply_extraction(response: Optional[str]) -> pd.Series:
            """Apply concept extraction to a single response"""
            try:
                if not response:
                    raise ValueError("Empty response")
                
                parsed_json = validate_json_response(response)
                if not parsed_json:
                    raise ValueError("Invalid JSON")
                
                concepts_by_level, methods = extract_concepts_and_methods(parsed_json)
                
                return pd.Series({
                    'level_0_concepts': concepts_by_level['level_0'],
                    'level_1_concepts': concepts_by_level['level_1'],
                    'level_2_concepts': concepts_by_level['level_2'],
                    'level_3_concepts': concepts_by_level['level_3'],
                    'level_4_concepts': concepts_by_level['level_4'],
                    'level_5_concepts': concepts_by_level['level_5'],
                    'methods': methods
                })
                
            except Exception as e:
                self.logger.warning(f"Error parsing concept response: {e}")
                return pd.Series({
                    'level_0_concepts': [],
                    'level_1_concepts': [],
                    'level_2_concepts': [],
                    'level_3_concepts': [],
                    'level_4_concepts': [],
                    'level_5_concepts': [],
                    'methods': []
                })
        
                                           
        extracted_concepts = concept_responses.apply(apply_extraction)
        
                                      
        result = {}
        for col in extracted_concepts.columns:
            result[col] = extracted_concepts[col].tolist()
            
        return result
    
    def get_unique_concepts_by_level(self, df: pd.DataFrame, 
                                   levels: List[int] = [0, 1, 2]) -> Dict[str, List[str]]:
        """
        Get unique concepts for specified levels
        
        Args:
            df: DataFrame with extracted concepts
            levels: List of levels to extract (default: [0, 1, 2])
            
        Returns:
            Dictionary with unique concepts for each level
        """
        unique_concepts = {}
        
        for level in levels:
            col_name = f'level_{level}_concepts'
            if col_name in df.columns:
                unique_concepts[f'level_{level}'] = (
                    df[col_name].explode().dropna().unique().tolist()
                )
            else:
                self.logger.warning(f"Column {col_name} not found in DataFrame")
                unique_concepts[f'level_{level}'] = []
        
        return unique_concepts
    
    def combine_concepts_for_clustering(self, df: pd.DataFrame, 
                                      levels: List[int] = [0, 1, 2]) -> List[str]:
        """
        Combine concepts from specified levels for clustering
        
        Args:
            df: DataFrame with extracted concepts
            levels: Levels to combine for clustering
            
        Returns:
            List of unique concepts from all specified levels
        """
        all_concepts = set()
        
        for level in levels:
            col_name = f'level_{level}_concepts'
            if col_name in df.columns:
                level_concepts = df[col_name].explode().dropna().unique()
                all_concepts.update(level_concepts)
        
        combined_concepts = list(all_concepts)
        self.logger.info(f"Combined {len(combined_concepts)} unique concepts from levels {levels}")
        
        return combined_concepts


def main():
    """Main function for running concept extractor standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract concepts from structured abstracts')
    parser.add_argument('--input', '-i', required=True, help='Input file path (structured abstracts)')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file path')
    parser.add_argument('--structured-col', default='structured_abs', help='Structured abstract column name')
    parser.add_argument('--combine-levels', nargs='+', type=int, default=[0, 1, 2],
                       help='Levels to combine for clustering (default: 0 1 2)')
    
    args = parser.parse_args()
    
               
    if args.input.endswith('.pkl'):
        df = pd.read_pickle(args.input)
    elif args.input.endswith('.json'):
        df = pd.read_json(args.input, lines=True)
    else:
        raise ValueError("Unsupported input format. Use PKL or JSON files.")
    
                          
    extractor = ConceptExtractor(args.config)
    
                      
    result_df = extractor.extract_concepts(df, args.structured_col)
    
                                  
    unique_concepts = extractor.get_unique_concepts_by_level(result_df, args.combine_levels)
    print(f"Unique concepts by level:")
    for level, concepts in unique_concepts.items():
        print(f"  {level}: {len(concepts)} concepts")
    
                                          
    combined_concepts = extractor.combine_concepts_for_clustering(result_df, args.combine_levels)
    print(f"Combined concepts for clustering: {len(combined_concepts)}")
    
                  
    if args.output:
        if args.output.endswith('.pkl'):
            result_df.to_pickle(args.output)
        elif args.output.endswith('.json'):
            result_df.to_json(args.output, orient='records', lines=True)
        else:
            raise ValueError("Output format must be .pkl or .json")
        print(f"Results saved to {args.output}")
    else:
        print(result_df[['c_list', 'level_0_concepts', 'level_1_concepts', 'level_2_concepts']].head())


if __name__ == "__main__":
    main()
