import pandas as pd
import scipy.sparse as sparse
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles preprocessing of implicit feedback data."""
    
    def __init__(self):
        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.user_map_reverse: Dict[int, str] = {}
        self.item_map_reverse: Dict[int, str] = {}
        self.popular_items: pd.Index = None
    
    def preprocess(self, df: pd.DataFrame, user_col: str, item_col: str, value_col: str) -> Tuple[sparse.csr_matrix, Dict]:
        """
        Preprocess implicit feedback data into a sparse user-item matrix.
        
        Args:
            df: DataFrame with user-item interactions
            user_col: Column name for users
            item_col: Column name for items
            value_col: Column name for interaction values (e.g., clicks)
        
        Returns:
            Tuple of (sparse user-item matrix, metadata dictionary)
        """
        logger.info("Preprocessing data...")
        
        # Map users and items to integer indices
        self.user_map = {user: idx for idx, user in enumerate(df[user_col].unique())}
        self.item_map = {item: idx for idx, item in enumerate(df[item_col].unique())}
        self.user_map_reverse = {idx: user for user, idx in self.user_map.items()}
        self.item_map_reverse = {idx: item for item, idx in self.item_map.items()}
        
        # Normalize interaction values
        scaler = MinMaxScaler()
        df = df.copy()
        df['normalized_value'] = scaler.fit_transform(df[[value_col]])
        
        # Create sparse matrix
        rows = df[user_col].map(self.user_map)
        cols = df[item_col].map(self.item_map)
        values = df['normalized_value']
        
        interaction_matrix = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_map), len(self.item_map))
        )
        
        # Store popular items for cold start
        self.popular_items = df.groupby(item_col)[value_col].sum().sort_values(ascending=False).index[:100]
        
        metadata = {
            'user_map': self.user_map,
            'item_map': self.item_map,
            'user_map_reverse': self.user_map_reverse,
            'item_map_reverse': self.item_map_reverse,
            'popular_items': self.popular_items
        }
        
        return interaction_matrix, metadata