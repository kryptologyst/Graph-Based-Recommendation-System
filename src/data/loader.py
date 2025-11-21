"""Data generation and loading utilities for graph-based recommendation system."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Generate synthetic user-item interaction data for graph-based recommendations."""
    
    def __init__(self, config: Dict):
        """Initialize data generator with configuration.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.random_state = np.random.RandomState(config.get('random_seed', 42))
        
    def generate_interactions(self, 
                            n_users: int = 1000, 
                            n_items: int = 500,
                            sparsity: float = 0.95) -> pd.DataFrame:
        """Generate synthetic user-item interactions with realistic patterns.
        
        Args:
            n_users: Number of users to generate
            n_items: Number of items to generate
            sparsity: Fraction of user-item pairs with no interaction
            
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        logger.info(f"Generating interactions for {n_users} users and {n_items} items")
        
        # Generate user and item IDs
        user_ids = [f"user_{i:04d}" for i in range(n_users)]
        item_ids = [f"item_{i:04d}" for i in range(n_items)]
        
        # Create popularity bias - some items are more popular
        item_popularity = self.random_state.power(2, n_items)
        item_popularity = item_popularity / item_popularity.sum()
        
        # Create user activity bias - some users are more active
        user_activity = self.random_state.power(1.5, n_users)
        user_activity = user_activity / user_activity.sum()
        
        interactions = []
        
        # Generate interactions based on popularity and activity
        for user_idx, user_id in enumerate(user_ids):
            # Number of interactions for this user
            n_interactions = int(self.random_state.poisson(20 * user_activity[user_idx]))
            n_interactions = max(5, min(n_interactions, n_items // 2))  # Reasonable bounds
            
            # Sample items based on popularity
            item_indices = self.random_state.choice(
                n_items, 
                size=n_interactions, 
                replace=False, 
                p=item_popularity
            )
            
            for item_idx in item_indices:
                item_id = item_ids[item_idx]
                
                # Generate rating (1-5) with some bias towards higher ratings
                rating = self.random_state.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                
                # Generate timestamp (recent dates)
                timestamp = self.random_state.uniform(0, 365)  # Days ago
                
                interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        df = pd.DataFrame(interactions)
        
        # Add some temporal patterns (seasonality)
        df['timestamp'] = pd.Timestamp.now() - pd.to_timedelta(df['timestamp'], unit='D')
        
        logger.info(f"Generated {len(df)} interactions")
        return df
    
    def generate_items(self, n_items: int = 500) -> pd.DataFrame:
        """Generate synthetic item metadata.
        
        Args:
            n_items: Number of items to generate
            
        Returns:
            DataFrame with item metadata
        """
        logger.info(f"Generating metadata for {n_items} items")
        
        # Categories for items
        categories = ['electronics', 'books', 'clothing', 'home', 'sports', 'beauty', 'toys', 'food']
        
        items = []
        for i in range(n_items):
            item_id = f"item_{i:04d}"
            category = self.random_state.choice(categories)
            
            # Generate title based on category
            titles = {
                'electronics': ['Smart Phone', 'Laptop', 'Headphones', 'Camera', 'Tablet'],
                'books': ['Novel', 'Textbook', 'Biography', 'Cookbook', 'Manual'],
                'clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes'],
                'home': ['Lamp', 'Chair', 'Table', 'Bed', 'Sofa'],
                'sports': ['Basketball', 'Tennis Racket', 'Running Shoes', 'Yoga Mat', 'Dumbbells'],
                'beauty': ['Lipstick', 'Foundation', 'Shampoo', 'Perfume', 'Moisturizer'],
                'toys': ['Action Figure', 'Board Game', 'Puzzle', 'Doll', 'Building Blocks'],
                'food': ['Chocolate', 'Coffee', 'Tea', 'Snacks', 'Cereal']
            }
            
            title = f"{self.random_state.choice(titles[category])} {i}"
            
            # Generate tags
            n_tags = self.random_state.randint(2, 6)
            all_tags = ['popular', 'trending', 'new', 'sale', 'premium', 'eco-friendly', 'durable', 'stylish']
            tags = self.random_state.choice(all_tags, size=n_tags, replace=False).tolist()
            
            items.append({
                'item_id': item_id,
                'title': title,
                'category': category,
                'tags': '|'.join(tags),
                'price': self.random_state.uniform(10, 500)
            })
        
        df = pd.DataFrame(items)
        logger.info(f"Generated metadata for {len(df)} items")
        return df
    
    def generate_users(self, n_users: int = 1000) -> pd.DataFrame:
        """Generate synthetic user metadata.
        
        Args:
            n_users: Number of users to generate
            
        Returns:
            DataFrame with user metadata
        """
        logger.info(f"Generating metadata for {n_users} users")
        
        users = []
        for i in range(n_users):
            user_id = f"user_{i:04d}"
            age = self.random_state.randint(18, 80)
            gender = self.random_state.choice(['M', 'F', 'Other'])
            
            # Generate location
            locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'CN']
            location = self.random_state.choice(locations)
            
            users.append({
                'user_id': user_id,
                'age': age,
                'gender': gender,
                'location': location,
                'signup_date': pd.Timestamp.now() - pd.Timedelta(days=self.random_state.randint(30, 1000))
            })
        
        df = pd.DataFrame(users)
        logger.info(f"Generated metadata for {len(df)} users")
        return df


class DataLoader:
    """Load and preprocess data for graph-based recommendation system."""
    
    def __init__(self, config: Dict):
        """Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.data_dir = Path(config['data']['interactions_file']).parent
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load interaction, item, and user data.
        
        Returns:
            Tuple of (interactions_df, items_df, users_df)
        """
        logger.info("Loading data files")
        
        # Load interactions
        interactions_path = self.data_dir / "interactions.csv"
        if not interactions_path.exists():
            logger.warning("Interactions file not found, generating synthetic data")
            generator = DataGenerator(self.config)
            interactions_df = generator.generate_interactions()
            interactions_df.to_csv(interactions_path, index=False)
        else:
            interactions_df = pd.read_csv(interactions_path)
        
        # Load items
        items_path = self.data_dir / "items.csv"
        if not items_path.exists():
            logger.warning("Items file not found, generating synthetic data")
            generator = DataGenerator(self.config)
            items_df = generator.generate_items()
            items_df.to_csv(items_path, index=False)
        else:
            items_df = pd.read_csv(items_path)
        
        # Load users
        users_path = self.data_dir / "users.csv"
        if not users_path.exists():
            logger.warning("Users file not found, generating synthetic data")
            generator = DataGenerator(self.config)
            users_df = generator.generate_users()
            users_df.to_csv(users_path, index=False)
        else:
            users_df = pd.read_csv(users_path)
        
        logger.info(f"Loaded {len(interactions_df)} interactions, {len(items_df)} items, {len(users_df)} users")
        return interactions_df, items_df, users_df
    
    def filter_data(self, 
                   interactions_df: pd.DataFrame,
                   items_df: pd.DataFrame,
                   users_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter data based on minimum interaction thresholds.
        
        Args:
            interactions_df: User-item interactions
            items_df: Item metadata
            users_df: User metadata
            
        Returns:
            Filtered DataFrames
        """
        logger.info("Filtering data based on interaction thresholds")
        
        min_user_interactions = self.config['data']['min_interactions_per_user']
        min_item_interactions = self.config['data']['min_interactions_per_item']
        
        # Filter users with minimum interactions
        user_counts = interactions_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        interactions_df = interactions_df[interactions_df['user_id'].isin(valid_users)]
        
        # Filter items with minimum interactions
        item_counts = interactions_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        interactions_df = interactions_df[interactions_df['item_id'].isin(valid_items)]
        
        # Filter metadata to only include valid users/items
        users_df = users_df[users_df['user_id'].isin(interactions_df['user_id'].unique())]
        items_df = items_df[items_df['item_id'].isin(interactions_df['item_id'].unique())]
        
        logger.info(f"After filtering: {len(interactions_df)} interactions, {len(items_df)} items, {len(users_df)} users")
        return interactions_df, items_df, users_df
    
    def create_train_test_split(self, interactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-aware train/validation/test splits.
        
        Args:
            interactions_df: User-item interactions with timestamps
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating time-aware train/validation/test splits")
        
        # Sort by timestamp
        interactions_df = interactions_df.sort_values('timestamp')
        
        # Split by time
        test_size = self.config['data']['test_size']
        val_size = self.config['data']['val_size']
        
        n_interactions = len(interactions_df)
        test_start = int(n_interactions * (1 - test_size))
        val_start = int(n_interactions * (1 - test_size - val_size))
        
        train_df = interactions_df.iloc[:val_start].copy()
        val_df = interactions_df.iloc[val_start:test_start].copy()
        test_df = interactions_df.iloc[test_start:].copy()
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Generate and save data
    generator = DataGenerator(config)
    
    interactions_df = generator.generate_interactions()
    items_df = generator.generate_items()
    users_df = generator.generate_users()
    
    # Save to data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    interactions_df.to_csv(data_dir / "interactions.csv", index=False)
    items_df.to_csv(data_dir / "items.csv", index=False)
    users_df.to_csv(data_dir / "users.csv", index=False)
    
    print("Data generation completed!")
    print(f"Generated {len(interactions_df)} interactions, {len(items_df)} items, {len(users_df)} users")
