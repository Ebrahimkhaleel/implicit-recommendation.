import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluates the recommender system using precision@k and recall@k."""
    
    def __init__(self, recommender):
        """
        Initialize the evaluator.
        
        Args:
            recommender: Instance of ImplicitRecommender
        """
        self.recommender = recommender
    
    def evaluate(self, test_data: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        """
        Evaluate the recommender system.
        
        Args:
            test_data: Test DataFrame with user-item interactions
            k: Number of recommendations to evaluate
        
        Returns:
            Dictionary with precision@k and recall@k
        """
        logger.info("Evaluating model...")
        precision_scores = []
        recall_scores = []
        
        for user_id in test_data['user_id'].unique():
            if user_id not in self.recommender.user_map:
                continue
                
            actual_items = set(test_data[test_data['user_id'] == user_id]['item_id'])
            actual_items = {item for item in actual_items if item in self.recommender.item_map}
            
            if not actual_items:
                continue
                
            recommendations = self.recommender.recommend(user_id, k)
            recommended_items = {item_id for item_id, _ in recommendations}
            
            hits = len(actual_items.intersection(recommended_items))
            precision = hits / k if k > 0 else 0
            recall = hits / len(actual_items) if len(actual_items) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        return {
            "precision@k": np.mean(precision_scores) if precision_scores else 0,
            "recall@k": np.mean(recall_scores) if recall_scores else 0
        }