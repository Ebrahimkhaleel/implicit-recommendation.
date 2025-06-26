from implicit.als import AlternatingLeastSquares
import scipy.sparse as sparse
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class ImplicitRecommender:
    """Recommendation system using implicit feedback with ALS."""
    
    def __init__(self, factors: int = 100, regularization: float = 0.01, iterations: int = 15):
        """
        Initialize the recommender system.
        
        Args:
            factors: Number of latent factors for ALS
            regularization: Regularization parameter for ALS
            iterations: Number of iterations for ALS training
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=False
        )
        self.user_map = {}
        self.item_map = {}
        self.user_map_reverse = {}
        self.item_map_reverse = {}
        self.popular_items = None
        self.interaction_matrix = None
    
    def set_metadata(self, metadata: Dict) -> None:
        """
        Set metadata from preprocessing.
        
        Args:
            metadata: Dictionary with user_map, item_map, user_map_reverse, item_map_reverse, popular_items
        """
        self.user_map = metadata['user_map']
        self.item_map = metadata['item_map']
        self.user_map_reverse = metadata['user_map_reverse']
        self.item_map_reverse = metadata['item_map_reverse']
        self.popular_items = metadata['popular_items']
    
    def train(self, interaction_matrix: sparse.csr_matrix) -> None:
        """
        Train the ALS model.
        
        Args:
            interaction_matrix: Sparse user-item interaction matrix
        """
        logger.info("Training ALS model...")
        self.interaction_matrix = interaction_matrix
        self.model.fit(interaction_matrix)
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_map:
            logger.warning(f"User {user_id} not found, using cold start")
            return self._cold_start_recommendations(n_recommendations)
        
        user_idx = self.user_map[user_id]
        recommended_items, scores = self.model.recommend(
            user_idx,
            self.interaction_matrix[user_idx],
            N=n_recommendations
        )
        
        return [(self.item_map_reverse[item_idx], float(score)) for item_idx, score in zip(recommended_items, scores)]
    
    def _cold_start_recommendations(self, n_recommendations: int) -> List[Tuple[str, float]]:
        """
        Handle cold start by recommending popular items.
        
        Returns:
            List of (item_id, score) tuples
        """
        return [(item_id, 1.0) for item_id in self.popular_items[:n_recommendations]]
    
    def explain_recommendation(self, user_id: str, item_id: str) -> Dict:
        """
        Explain why an item was recommended for a user.
        
        Args:
            user_id: User ID
            item_id: Item ID
        
        Returns:
            Dictionary with explanation details
        """
        if user_id not in self.user_map or item_id not in self.item_map:
            return {"explanation": "Cold start: Popular item recommendation"}
        
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]
        
        # Get similar items (returns two arrays: indices and scores)
        similar_item_indices, similar_item_scores = self.model.similar_items(item_idx, N=5)
        explanation = {
            "explanation": f"Recommended because user interacted with similar items",
            "similar_items": [
                {"item_id": self.item_map_reverse[idx], "similarity": float(score)}
                for idx, score in zip(similar_item_indices, similar_item_scores)
            ]
        }
        
        return explanation