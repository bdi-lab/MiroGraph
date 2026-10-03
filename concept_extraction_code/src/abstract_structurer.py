"""
Abstract Structurer Module
Converts unstructured paper abstracts into structured format using LLM
"""

import json
import logging
import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path

from .utils import Config, LLMClient, PromptLoader, setup_logging, validate_json_response, save_intermediate_result


class AbstractStructurer:
    """Class for structuring paper abstracts using LLM"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.client = LLMClient(self.config)
        self.prompt_loader = PromptLoader(self.config)
        
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
                                
        self.prompt = self.prompt_loader.load_prompt(
            self.config.get('prompts.structurizer', 'abstract_structuring.txt')
        )
        
    def format_input_text(self, row: pd.Series) -> str:
        """Format paper data for LLM input"""
        return f"""TITLE: {row['title']}
        ABSTRACT: {row['abstract']}
        """
    
    def structure_abstracts(self, df: pd.DataFrame, 
                          title_col: str = 'title', 
                          abstract_col: str = 'abstract') -> pd.DataFrame:
        """
        Structure abstracts in the dataframe
        
        Args:
            df: DataFrame with paper data
            title_col: Column name for paper titles
            abstract_col: Column name for abstracts
            
        Returns:
            DataFrame with structured abstracts and validation status
        """
        self.logger.info(f"Starting abstract structuring for {len(df)} papers")
        
                                          
        df['input_text'] = df.apply(
            lambda row: self.format_input_text(row), axis=1
        )
        
                                           
        structured_responses = self.client.process_batch_parallel(
            self.prompt, df['input_text'].tolist()
        )
        
        df['structured_abs'] = structured_responses
        
                                     
        self._extract_validation_status(df)
        
                                                 
        if self.config.get('processing.save_intermediate_results', True):
            output_dir = Path(self.config.get('paths.output_dir', './output'))
            output_dir.mkdir(parents=True, exist_ok=True)
            save_intermediate_result(
                df, 
                output_dir / 'structured_abstracts.pkl',
                'structured abstracts'
            )
        
        valid_count = df['status'].value_counts().get('valid', 0)
        self.logger.info(f"Abstract structuring completed. Valid papers: {valid_count}/{len(df)}")
        
        return df
    
    def _extract_validation_status(self, df: pd.DataFrame) -> None:
        """Extract validation status from structured abstracts"""
        def get_status(response: Optional[str]) -> str:
            if not response:
                return 'invalid'
            
            parsed = validate_json_response(response)
            if not parsed:
                return 'invalid'
            
            return parsed.get('status', 'invalid')
        
        df['status'] = df['structured_abs'].apply(get_status)
        
    def get_valid_papers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter and return only valid structured papers"""
        if 'status' not in df.columns:
            raise ValueError("DataFrame must have 'status' column. Run structure_abstracts first.")
        
        valid_df = df[df['status'] == 'valid'].copy()
        
                                                                              
        def clean_structured_abs(abs_str: Optional[str]) -> Optional[str]:
            if not abs_str:
                return abs_str
            return abs_str.replace('\\n  "status": "valid",', '')
        
        valid_df['structured_abs'] = valid_df['structured_abs'].apply(clean_structured_abs)
        
        self.logger.info(f"Filtered to {len(valid_df)} valid papers")
        return valid_df


def main():
    """Main function for running abstract structurer standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Structure paper abstracts using LLM')
    parser.add_argument('--input', '-i', required=True, help='Input file path (JSON or pickle)')
    parser.add_argument('--output', '-o', help='Output file path (optional)')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file path')
    parser.add_argument('--title-col', default='title', help='Title column name')
    parser.add_argument('--abstract-col', default='abstract', help='Abstract column name')
    parser.add_argument('--valid-only', action='store_true', help='Output only valid papers')
    
    args = parser.parse_args()
    
               
    if args.input.endswith('.json') or args.input.endswith('.gz'):
        df = pd.read_json(args.input, compression='gzip' if args.input.endswith('.gz') else None, lines=True)
    elif args.input.endswith('.pkl'):
        df = pd.read_pickle(args.input)
    else:
        raise ValueError("Unsupported input format. Use JSON, JSON.gz, or PKL files.")
    
                           
    structurer = AbstractStructurer(args.config)
    
                         
    result_df = structurer.structure_abstracts(
        df, 
        title_col=args.title_col, 
        abstract_col=args.abstract_col
    )
    
                                         
    if args.valid_only:
        result_df = structurer.get_valid_papers(result_df)
    
                  
    if args.output:
        if args.output.endswith('.pkl'):
            result_df.to_pickle(args.output)
        elif args.output.endswith('.json'):
            result_df.to_json(args.output, orient='records', lines=True)
        else:
            raise ValueError("Output format must be .pkl or .json")
        print(f"Results saved to {args.output}")
    else:
        print(result_df.head())


if __name__ == "__main__":
    main()
