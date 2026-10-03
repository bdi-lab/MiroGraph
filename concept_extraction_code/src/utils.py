"""
Utility functions and classes for the LLM Prompting Pipeline
"""

import os
import json
import time
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI, APIError, RateLimitError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


@dataclass
class Config:
    """Configuration class for the pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'api.openai_api_key')"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from config or environment"""
        api_key = self.get('api.openai_api_key')
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
        return api_key


class LLMClient:
    """OpenAI API client wrapper with retry logic"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model_name = config.get('api.model_name', 'gpt-4o-mini')
        self.temperature = config.get('api.temperature', 0)
        self.top_p = config.get('api.top_p', 0)
        self.max_retries = config.get('api.max_retries', 5)
        self.use_responses_api = config.get('api.use_responses_api', None)

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if not lines:
            return stripped

        first = lines[0].strip()
        last = lines[-1].strip()

        if not first.startswith("```"):
            return stripped
        if last != "```":
            return stripped

        return "\n".join(lines[1:-1]).strip()

    def _request_via_responses_api(self, prompt: str, input_text: str) -> Optional[str]:
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
        )

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        try:
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) != "message":
                    continue
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if isinstance(text, str) and text.strip():
                        return text.strip()
        except Exception:
            return None

        return None

    def _request_via_chat_completions(self, prompt: str, input_text: str) -> Optional[str]:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
        )
        content = completion.choices[0].message.content
        return content.strip() if isinstance(content, str) else None
        
    def get_response_with_retry(self, prompt: str, input_text: str) -> Optional[str]:
        """Get response from OpenAI with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if self.use_responses_api is False:
                    raw_text = self._request_via_chat_completions(prompt, input_text)
                else:
                    try:
                        raw_text = self._request_via_responses_api(prompt, input_text)
                    except Exception:
                        if self.use_responses_api is True:
                            raise
                        raw_text = self._request_via_chat_completions(prompt, input_text)

                if not raw_text:
                    return None

                return self._strip_markdown_code_fence(raw_text)
                
            except RateLimitError as e:
                wait_time = 2 ** attempt                       
                logging.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
            except APIError as e:
                logging.error(f"API Error: {e}. Retrying...")
                time.sleep(2)
                
        logging.error("Failed after maximum retries.")
        return None
        
    def process_batch_parallel(self, prompt: str, input_texts: List[str], 
                             max_workers: Optional[int] = None) -> List[Optional[str]]:
        """Process multiple inputs in parallel"""
        if max_workers is None:
            max_workers = self.config.get('api.max_workers', 100)
            
        results = [None] * len(input_texts)
        
        def process_single(idx_text_pair):
            idx, text = idx_text_pair
            try:
                return idx, self.get_response_with_retry(prompt, text)
            except Exception as e:
                logging.error(f"Error processing item {idx}: {e}")
                return idx, None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single, (idx, text)): idx
                for idx, text in enumerate(input_texts)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                idx, result = future.result()
                results[idx] = result
                
        return results


class PromptLoader:
    """Load prompts from files"""
    
    def __init__(self, config: Config):
        prompt_dir = Path(config.get('paths.prompts_dir', 'config/prompts'))
        if prompt_dir.is_absolute():
            self.prompts_dir = prompt_dir
            return

        candidates = [
            Path.cwd() / prompt_dir,
            config.config_path.parent / prompt_dir,
            config.config_path.parent.parent / prompt_dir,
        ]
        for candidate in candidates:
            if candidate.exists():
                self.prompts_dir = candidate
                return
        self.prompts_dir = candidates[0]
        
    def load_prompt(self, prompt_name: str) -> str:
        """Load prompt from file"""
        prompt_path = self.prompts_dir / prompt_name
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()


def get_tqdm(config: Config, *args, **kwargs):
    """Get tqdm progress bar based on config settings"""
    show_progress = config.get('processing.show_progress_bars', True)
    if not show_progress:
        kwargs['disable'] = True
    return tqdm(*args, **kwargs)


def setup_logging(config: Config) -> None:
    """Setup logging configuration"""
    log_level = getattr(logging, config.get('processing.log_level', 'INFO'))
    suppress_third_party = config.get('processing.suppress_third_party_logs', True)
    
                           
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
                                                   
    if suppress_third_party:
        third_party_loggers = [
            'httpx', 'openai', 'urllib3', 'requests', 'transformers', 
            'sentence_transformers', 'torch', 'sklearn', 'adapters'
        ]
        for logger_name in third_party_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
                                                       
    app_loggers = [
        '__main__', 'src.abstract_structurer', 'src.concept_extractor', 
        'src.concept_clustering', 'src.cluster_validator', 'src.utils'
    ]
    for logger_name in app_loggers:
        logging.getLogger(logger_name).setLevel(log_level)


def validate_json_response(response: str) -> Optional[Dict]:
    """Validate and parse JSON response"""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return None


def save_intermediate_result(data: Any, filepath: Union[str, Path], description: str = "") -> None:
    """Save intermediate results"""
    try:
                                                 
        filepath_str = str(filepath) if hasattr(filepath, '__fspath__') or isinstance(filepath, Path) else filepath
        
        if filepath_str.endswith('.pkl'):
            import pickle
            with open(filepath_str, 'wb') as f:
                pickle.dump(data, f)
        elif filepath_str.endswith('.json'):
            with open(filepath_str, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
                                     
            if hasattr(data, 'to_pickle'):
                data.to_pickle(filepath_str)
            else:
                raise ValueError(f"Unsupported file format: {filepath_str}")
                
        logging.info(f"Saved {description} to {filepath_str}")
        
    except Exception as e:
        logging.error(f"Failed to save {description} to {filepath}: {e}")
