"""
LLM-based Cluster Validation and Partition Module
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from utils import Config, LLMClient, PromptLoader


class LLMClusterValidator:
    """LLM-based cluster validation and partition"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.client = LLMClient(self.config)
        self.prompt_loader = PromptLoader(self.config)
        self.logger = logging.getLogger(__name__)
        self.fail_on_error = self.config.get('llm.fail_on_error', False)
        
        # Load prompt (use default prompt if not found)
        try:
            self.validation_prompt = self.prompt_loader.load_prompt(
                self.config.get('prompts.cluster_validator', 'prompt_cluster_validation_v2.txt')
            )
        except:
            self.validation_prompt = self._get_default_validation_prompt()
    
    def _get_default_validation_prompt(self) -> str:
        """Default validation prompt"""
        return """You are an expert in semantic analysis. Your task is to evaluate whether a set of keywords share similar semantic meaning.

Given a list of keywords, determine if they are semantically similar. If they are NOT semantically similar, partition them into semantically coherent groups.

Use a strict, fine-grained standard:
- Do not treat a cluster as coherent only because many keywords contain generic words such as "evaluation", "benchmark", "dataset", "performance", "experiment", "application", "method", or "model".
- Do not group broad application/domain labels such as education, politics, retail, government, entertainment, healthcare, news, or finance unless the keywords describe the same concrete research problem.
- Prefer smaller partitions around concrete tasks, methods, model families, data structures, metrics, or problem settings.
- If a keyword is too generic to belong with the concrete group, put it in its own partition.

Return your response as a JSON object with the following structure:
{
    "are_similar": true/false,
    "reasoning": "brief explanation",
    "partitions": [
        ["keyword1", "keyword2", ...],
        ["keyword3", "keyword4", ...]
    ]
}

If keywords are similar, set "are_similar" to true and "partitions" should contain a single list with all keywords.
If keywords are NOT similar, set "are_similar" to false and "partitions" should contain multiple lists, each representing a semantically coherent group."""
    
    def validate_cluster(self, keywords: List[str]) -> Tuple[bool, List[List[str]]]:
        """
        Validate if keywords in a single cluster are semantically similar
        
        Args:
            keywords: List of keywords in the cluster
            
        Returns:
            (are_similar, partitions)
            - are_similar: Whether keywords are similar
            - partitions: List of partitioned keyword groups (single group if similar)
        """
        if len(keywords) == 0:
            return True, [keywords]
        
        if len(keywords) == 1:
            return True, [keywords]
        
        # Construct prompt
        keywords_str = "\n".join([f"- {kw}" for kw in keywords])
        input_text = f"""Keywords to evaluate:
{keywords_str}

Please evaluate if these keywords are semantically similar and partition them if necessary."""
        
        # Call LLM
        response = self.client.get_response_with_retry(self.validation_prompt, input_text)
        
        if response is None:
            message = "LLM response failed during cluster validation"
            if self.fail_on_error:
                raise RuntimeError(message)
            self.logger.warning(f"{message}. Returning default (assuming similar)")
            return True, [keywords]
        
        if not response or len(response.strip()) == 0:
            message = "LLM returned empty response during cluster validation"
            if self.fail_on_error:
                raise RuntimeError(message)
            self.logger.warning(f"{message}. Returning default (assuming similar)")
            return True, [keywords]
        
        try:
            result = json.loads(response)
            are_similar = result.get('are_similar', True)
            partitions = result.get('partitions', [keywords])
            
            # If partitions is empty or incorrectly formatted
            if not partitions or not isinstance(partitions, list):
                self.logger.warning(f"Invalid partitions format. Returning default")
                return True, [keywords]
            
            self.logger.debug(f"Validation result: similar={are_similar}, partitions={len(partitions)}")
            return are_similar, partitions
            
        except json.JSONDecodeError as e:
            message = f"JSON parsing error during cluster validation: {e}"
            self.logger.error(message)
            self.logger.warning(f"Response content (first 200 chars): {response[:200] if response else 'None'}")
            if self.fail_on_error:
                raise RuntimeError(message)
            return True, [keywords]
    
    def validate_all_clusters(self, cluster_to_concepts: Dict[int, List[str]],
                              max_cluster_size: int = 50) -> Dict[int, List[List[str]]]:
        """
        Validate and partition all clusters
        
        Args:
            cluster_to_concepts: {cluster_id: [concept1, ...]} dictionary
            max_cluster_size: Maximum cluster size to skip validation
            
        Returns:
            {cluster_id: [[concept1, ...], [concept2, ...], ...]} dictionary
            (partitioned result)
        """
        self.logger.info(f"Cluster validation started: {len(cluster_to_concepts)} clusters")
        
        validated_clusters = {}
        
        for cluster_id, concepts in tqdm(cluster_to_concepts.items(), desc="Cluster validation"):
            if len(concepts) > max_cluster_size:
                self.logger.info(f"Cluster {cluster_id}: Too large ({len(concepts)}). Skipping validation")
                validated_clusters[cluster_id] = [concepts]
                continue
            
            are_similar, partitions = self.validate_cluster(concepts)
            
            if are_similar:
                self.logger.info(f"Cluster {cluster_id}: Similar ({len(concepts)} keywords)")
                validated_clusters[cluster_id] = [concepts]
            else:
                self.logger.info(f"Cluster {cluster_id}: Not similar. Partitioned into {len(partitions)} groups")
                validated_clusters[cluster_id] = partitions
        
        total_partitions = sum(len(partitions) for partitions in validated_clusters.values())
        self.logger.info(f"Validation completed: {total_partitions} final clusters")
        
        return validated_clusters
