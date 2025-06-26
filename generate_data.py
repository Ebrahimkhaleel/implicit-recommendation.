import pandas as pd
import numpy as np
import uuid
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_implicit_feedback_data(
    n_users: int = 1000,
    n_items: int = 5000,
    n_interactions: int = 100000,
    output_file: str = "implicit_feedback_data.csv"
) -> pd.DataFrame:
    """
    Generate synthetic implicit feedback data for a recommender system.
    
    Args:
        n_users: Number of unique users
        n_items: Number of unique items
        n_interactions: Number of user-item interactions
        output_file: Path to save the generated CSV file
    
    Returns:
        DataFrame with columns: user_id, item_id, clicks
    """
    logger.info("Generating synthetic implicit feedback data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate user IDs (UUIDs truncated for simplicity)
    user_ids = [str(uuid.uuid4())[:8] for _ in range(n_users)]
    
    # Generate item IDs
    item_ids = [f"item_{i}" for i in range(n_items)]
    
    # Simulate item popularity with a power-law distribution
    # Popular items get more interactions
    item_popularity = np.random.zipf(a=2, size=n_items)  # Zipf distribution for popularity
    item_weights = item_popularity / item_popularity.sum()  # Normalize to probabilities
    
    # Generate interactions
    data = {
        'user_id': np.random.choice(user_ids, size=n_interactions, replace=True),
        'item_id': np.random.choice(item_ids, size=n_interactions, replace=True, p=item_weights),
        'clicks': np.random.randint(1, 10, size=n_interactions)  # Random click counts
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Remove duplicates by aggregating clicks for the same user-item pair
    df = df.groupby(['user_id', 'item_id'])['clicks'].sum().reset_index()
    
    # Log dataset statistics
    logger.info(f"Generated dataset with {len(df)} unique interactions")
    logger.info(f"Number of users: {df['user_id'].nunique()}")
    logger.info(f"Number of items: {df['item_id'].nunique()}")
    logger.info(f"Sparsity: {len(df) / (n_users * n_items):.4%}")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Data saved to {output_file}")
    
    return df

def main():
    """Generate and save synthetic implicit feedback data."""
    output_file = "implicit_feedback_data.csv"
    df = generate_implicit_feedback_data(
        n_users=1000,
        n_items=5000,
        n_interactions=10000,
        output_file=output_file
    )
    logger.info(f"Sample data:\n{df.head()}")

if __name__ == "__main__":
    main()