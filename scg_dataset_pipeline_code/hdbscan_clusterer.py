"""
HDBSCAN Probability-based Clustering Module
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path

try:
    import hdbscan
    from hdbscan.prediction import membership_vector
except ImportError:
    raise ImportError("hdbscan is not installed. Please install it with 'pip install hdbscan'.")


class HDBSCANProbabilityClusterer:
    """Probability-based HDBSCAN clustering"""
    
    def __init__(self,
                 min_cluster_size: int = 2,
                 min_samples: Optional[int] = None,
                 metric: str = 'euclidean',
                 cluster_selection_method: str = 'leaf',
                 cluster_selection_epsilon: float = 0.1,
                 prediction_data: bool = True,
                 allow_single_cluster: bool = False):
        """
        Args:
            min_cluster_size: Minimum cluster size (default: 2 to reduce outliers)
                - Smaller values allow smaller clusters, reducing outliers
                - Minimum value is 2 (HDBSCAN requirement)
            min_samples: Minimum number of samples for cluster core points (default: 1 to reduce outliers)
                - Smaller values make more points eligible as core points, reducing outliers
                - Typically set to 1 or same as min_cluster_size
            metric: Distance metric ('euclidean' recommended, use L2 normalized vectors)
            cluster_selection_method: Cluster selection method ('eom', 'leaf')
                - 'leaf': Selects leaf clusters in the tree (more aggressive, reduces outliers)
                - 'eom': Excess of Mass method (more conservative)
            cluster_selection_epsilon: Epsilon value for cluster selection (default: 0.1)
                - Higher values merge more clusters and reduce outliers
                - Range: 0.0 to 1.0, typically 0.0-0.5
                - To minimize outliers, try: 0.2, 0.3, or 0.5
            prediction_data: Whether to generate prediction data (required for probability calculation)
            allow_single_cluster: Allow all points to form a single cluster (default: False)
                - If True, can eliminate all outliers but may create one giant cluster
        
        HDBSCAN Outlier Detection:
        - HDBSCAN uses density-based clustering: points in low-density regions are outliers (label = -1)
        - Outliers are determined by:
          1. Not having enough neighbors within the density threshold
          2. Not being part of any cluster that meets min_cluster_size
          3. Being in sparse regions of the embedding space
        - To minimize outliers:
          1. Decrease min_cluster_size (already at minimum 2)
          2. Decrease min_samples (already at minimum 1)
          3. Use 'leaf' method (already set)
          4. Increase cluster_selection_epsilon (try 0.2-0.5)
          5. Set allow_single_cluster=True (extreme: eliminates all outliers)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else 1  # Default to 1 to reduce outliers
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.prediction_data = prediction_data
        self.allow_single_cluster = allow_single_cluster
        
        self.logger = logging.getLogger(__name__)
        self.clusterer = None
        self.labels_ = None
        
    def fit(self, embeddings: np.ndarray) -> 'HDBSCANProbabilityClusterer':
        """
        Perform HDBSCAN clustering on embeddings
        
        Args:
            embeddings: Embedding array of shape (n_samples, n_features)
            
        Returns:
            self
        """
        self.logger.info(f"HDBSCAN clustering started: {len(embeddings)} data points")
        
        # L2 normalization (preparation for cosine distance)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            prediction_data=self.prediction_data,
            allow_single_cluster=self.allow_single_cluster
        )
        
        self.clusterer.fit(normalized_embeddings)
        self.labels_ = self.clusterer.labels_
        
        # Statistics
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        
        self.logger.info(f"Clustering completed:")
        self.logger.info(f"  - Number of clusters created: {n_clusters}")
        self.logger.info(f"  - Number of noise points: {n_noise} ({n_noise/len(embeddings)*100:.2f}%)")
        
        return self
    
    def get_outlier_indices(self) -> List[int]:
        """
        Return indices of keywords classified as outliers
        
        Returns:
            List of outlier indices
        """
        if self.labels_ is None:
            raise ValueError("Clustering has not been performed. Call fit() first.")
        
        outlier_indices = np.where(self.labels_ == -1)[0].tolist()
        return outlier_indices
    
    def get_membership_probabilities(self,
                                     embeddings: np.ndarray = None,
                                     method: str = 'exact',
                                     batch_size: int = 128,
                                     cache_dir: str = None) -> np.ndarray:
        """
        Calculate probability of each keyword belonging to each cluster
        
        Args:
            embeddings: Embedding array (optional, returns probabilities for all data used in fit)
            
        Returns:
            Probability array of shape (n_samples, n_clusters) (based on all data used in fit)
        """
        if self.clusterer is None or not self.prediction_data:
            raise ValueError("Clustering must be performed with prediction_data=True to calculate probabilities.")
        
        method = (method or 'exact').lower()

        # Calculate cluster membership probabilities for all data used in fit
        # This can be slow for large datasets - add logging
        n_samples = len(self.labels_)
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.logger.info(
            f"Calculating membership probabilities for {n_samples:,} samples across "
            f"{n_clusters:,} clusters (method={method})..."
        )

        if method == 'exact':
            self.logger.info("Exact all-points method has no progress bar and may take a long time.")
            membership_probs = hdbscan.all_points_membership_vectors(self.clusterer)
        elif method == 'batch':
            if embeddings is None:
                embeddings = self.clusterer.prediction_data_.raw_data
            batch_size = max(1, int(batch_size))
            cache_path = Path(cache_dir) if cache_dir else None
            if cache_path:
                cache_path.mkdir(parents=True, exist_ok=True)

            chunks = []
            starts = list(range(0, len(embeddings), batch_size))
            for batch_idx, start in enumerate(tqdm(
                starts,
                total=len(starts),
                desc="Membership probabilities",
                dynamic_ncols=True,
                unit="batch"
            )):
                end = min(start + batch_size, len(embeddings))
                batch_file = cache_path / f"batch_{batch_idx:05d}_{start}_{end}.npy" if cache_path else None
                if batch_file and batch_file.exists():
                    chunks.append(np.load(batch_file))
                    continue

                batch = embeddings[start:start + batch_size]
                batch_probs = membership_vector(self.clusterer, batch)
                if batch_file:
                    np.save(batch_file, batch_probs)
                chunks.append(batch_probs)
            membership_probs = np.vstack(chunks) if chunks else np.empty((0, n_clusters))
        else:
            raise ValueError(f"Unsupported probability assignment method: {method}")
        
        self.logger.info(f"Probability calculation completed: {membership_probs.shape}")
        
        return membership_probs
    
    def assign_concepts_to_clusters(self, 
                                   concepts: List[str],
                                   embeddings: np.ndarray,
                                   probability_threshold: float = 0.8,
                                   original_indices: List[int] = None,
                                   probability_method: str = 'exact',
                                   probability_batch_size: int = 128,
                                   all_embeddings: np.ndarray = None,
                                   probability_cache_dir: str = None) -> Tuple[Dict[int, List[str]], Dict[str, Dict[int, float]]]:
        """
        Assign keywords to clusters based on probability
        (duplicate assignment to all clusters with probability >= threshold * max_probability)
        
        Args:
            concepts: List of concept keywords (already outlier-removed)
            embeddings: Embedding array (already outlier-removed)
            probability_threshold: Threshold multiplier of max probability (e.g., 0.8 means 80% of max probability)
            original_indices: Indices of concepts in original data (assumed sequential from 0 if None)
            
        Returns:
            Tuple of:
            - {cluster_id: [concept1, concept2, ...]} dictionary
            - {concept: {cluster_id: probability, ...}} dictionary (concept to cluster probabilities)
        """
        if self.labels_ is None:
            raise ValueError("Clustering has not been performed. Call fit() first.")
        
        self.logger.info(f"Probability-based cluster assignment started (threshold: {probability_threshold})")
        self.logger.info(f"Processing {len(concepts):,} concepts...")
        
        # If original_indices is None, assume sequential from 0
        if original_indices is None:
            # If concepts is already outlier-removed list, find non-outlier indices from all data used in fit
            self.logger.info("Determining original indices for concepts...")
            outlier_indices = set(self.get_outlier_indices())
            all_indices = [i for i in range(len(self.labels_)) if i not in outlier_indices]
            original_indices = all_indices[:len(concepts)]
            self.logger.info(f"Found {len(original_indices):,} valid indices")
        
        # Calculate probabilities for all data used in fit
        # This is the bottleneck - can take 10-30 minutes for large datasets
        self.logger.info("Starting probability calculation (this is the slowest step)...")
        normalized_embeddings = None
        if probability_method == 'batch':
            source_embeddings = all_embeddings if all_embeddings is not None else embeddings
            norms = np.linalg.norm(source_embeddings, axis=1, keepdims=True)
            normalized_embeddings = source_embeddings / (norms + 1e-8)

        membership_probs = self.get_membership_probabilities(
            embeddings=normalized_embeddings,
            method=probability_method,
            batch_size=probability_batch_size,
            cache_dir=probability_cache_dir
        )
        self.logger.info("Probability calculation completed, starting cluster assignment...")
        
        # Cluster ID list (excluding noise)
        unique_labels = sorted([l for l in set(self.labels_) if l != -1])
        n_clusters = len(unique_labels)
        
        # Cluster assignment
        cluster_to_concepts = {cluster_id: [] for cluster_id in unique_labels}
        concept_to_cluster_probs = {}  # {concept: {cluster_id: probability, ...}}
        unassigned_concepts = []
        
        for i, orig_idx in enumerate(tqdm(original_indices, desc="Cluster assignment")):
            concept = concepts[i]
            # Use probabilities for original index
            probs = membership_probs[orig_idx]
            
            # Extract only cluster probabilities (excluding noise cluster)
            cluster_probs = probs[:n_clusters] if len(probs) >= n_clusters else probs
            
            # Find max probability
            max_prob = np.max(cluster_probs) if len(cluster_probs) > 0 else 0
            
            # Store probabilities for this concept
            concept_to_cluster_probs[concept] = {}
            for cluster_idx, prob in enumerate(cluster_probs):
                if cluster_idx < len(unique_labels):
                    cluster_id = unique_labels[cluster_idx]
                    concept_to_cluster_probs[concept][cluster_id] = float(prob)
            
            # If max_prob is 0, assign to the first available cluster (should not happen in practice)
            if max_prob == 0:
                if len(unique_labels) > 0:
                    # Assign to first cluster as fallback
                    cluster_id = unique_labels[0]
                    cluster_to_concepts[cluster_id].append(concept)
                    self.logger.warning(f"Concept '{concept}' has zero probability for all clusters, assigned to cluster {cluster_id} as fallback")
                else:
                    unassigned_concepts.append(concept)
                    self.logger.error(f"Concept '{concept}' cannot be assigned: no clusters available and max_prob is 0")
                continue
            
            # Assign to clusters with probability >= threshold * max_probability (allows duplicate assignment)
            threshold = max_prob * probability_threshold
            assigned_cluster_indices = np.where(cluster_probs >= threshold)[0]
            
            if len(assigned_cluster_indices) == 0:
                # If no cluster meets threshold, assign to the cluster with max probability anyway
                # This ensures all valid concepts get assigned
                max_prob_idx = np.argmax(cluster_probs)
                if max_prob_idx < n_clusters and max_prob_idx < len(unique_labels):
                    cluster_id = unique_labels[max_prob_idx]
                    cluster_to_concepts[cluster_id].append(concept)
                else:
                    # Fallback: assign to first cluster if argmax fails
                    if len(unique_labels) > 0:
                        cluster_id = unique_labels[0]
                        cluster_to_concepts[cluster_id].append(concept)
                        self.logger.warning(f"Concept '{concept}' argmax failed, assigned to first cluster {cluster_id} as fallback")
                    else:
                        unassigned_concepts.append(concept)
                        self.logger.error(f"Concept '{concept}' cannot be assigned: no clusters available")
            else:
                for prob_idx in assigned_cluster_indices:
                    if prob_idx < n_clusters:
                        cluster_id = unique_labels[prob_idx]
                        cluster_to_concepts[cluster_id].append(concept)
        
        # Remove empty clusters
        cluster_to_concepts = {k: v for k, v in cluster_to_concepts.items() if len(v) > 0}
        
        self.logger.info(f"Cluster assignment completed: {len(cluster_to_concepts)} clusters")
        self.logger.info(f"  - Assigned concepts: {sum(len(v) for v in cluster_to_concepts.values())}")
        self.logger.info(f"  - Unassigned concepts: {len(unassigned_concepts)}")
        for cluster_id, concept_list in cluster_to_concepts.items():
            self.logger.info(f"  - Cluster {cluster_id}: {len(concept_list)} keywords")
        
        return cluster_to_concepts, concept_to_cluster_probs
