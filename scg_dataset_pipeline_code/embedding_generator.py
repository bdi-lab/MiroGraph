"""
SPECTER2 Embedding Generation Module
"""

import numpy as np
import logging
import torch
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


class EmbeddingGenerator:
    """SPECTER2-based embedding generation"""
    
    def __init__(self, model_name: str = 'allenai/specter2_base',
                 batch_size: int = 50,
                 max_length: int = 512):
        """
        Args:
            model_name: SPECTER2 model name
            batch_size: Batch size
            max_length: Maximum token length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
        self.embedding_model = None
        self.tokenizer = None
        self.device = None
        
    def _load_model(self):
        """Load SPECTER2 model"""
        if self.embedding_model is not None:
            return
        
        self.logger.info(f"Loading SPECTER2 model: {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embedding_model = AutoAdapterModel.from_pretrained(self.model_name)
        
        # Load adapter for SPECTER2
        self.embedding_model.load_adapter("allenai/specter2", source="hf", set_active=True)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        self.logger.info(f"Model loaded (device: {self.device})")
    
    def generate_embeddings(self, concepts: List[str]) -> np.ndarray:
        """
        Generate embeddings for concept keywords
        
        Args:
            concepts: List of concept keywords
            
        Returns:
            Embedding array of shape (n_concepts, embedding_dim)
        """
        if self.embedding_model is None:
            self._load_model()
        
        self.logger.info(f"Generating embeddings for {len(concepts)} concepts...")
        
        dataloader = DataLoader(concepts, batch_size=self.batch_size)
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                output = self.embedding_model(**inputs)
                embeddings = output.last_hidden_state[:, 0, :]  # CLS token
                all_embeddings.append(embeddings.cpu())
        
        result = torch.cat(all_embeddings, dim=0).numpy()
        
        self.logger.info(f"Embedding generation completed: {result.shape}")
        
        return result

