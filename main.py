import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from preprocessing import DataPreprocessor
from recommender import ImplicitRecommender
from evaluation import Evaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Example usage of the implicit feedback recommender system."""
    # Load generated data
    df = pd.read_csv("implicit_feedback_data.csv")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    interaction_matrix, metadata = preprocessor.preprocess(train_df, 'user_id', 'item_id', 'clicks')
    
    # Initialize and train recommender
    recommender = ImplicitRecommender(factors=50, iterations=10)
    recommender.set_metadata(metadata)
    recommender.train(interaction_matrix)
    
    # Generate recommendations for a sample user
    sample_user = train_df['user_id'].iloc[0]
    recommendations = recommender.recommend(sample_user, n_recommendations=5)
    logger.info(f"Recommendations for user {sample_user}: {recommendations}")
    
    # Explain a recommendation
    sample_item = recommendations[0][0] if recommendations else recommender.popular_items[0]
    explanation = recommender.explain_recommendation(sample_user, sample_item)
    logger.info(f"Explanation for item {sample_item}: {explanation}")
    
    # Evaluate model
    evaluator = Evaluator(recommender)
    metrics = evaluator.evaluate(test_df, k=5)
    logger.info(f"Model performance: {metrics}")
    
    # For debugging: inspect similar items
    item_id = recommendations[0][0] if recommendations else recommender.popular_items[0]
    if item_id in recommender.item_map:
        item_idx = recommender.item_map[item_id]
        logger.info(f"Inspecting similar items for item ID {item_id}")
        similar_item_indices, similar_item_scores = recommender.model.similar_items(item_idx, N=10)
        for idx, score in zip(similar_item_indices, similar_item_scores):
            logger.info(f"Item ID {recommender.item_map_reverse[idx]} with score {score}")
    else:
        logger.warning(f"Item ID {item_id} not found in item_map for debugging")

if __name__ == "__main__":
    main()