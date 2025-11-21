#!/usr/bin/env python3
"""Simple test script to verify the graph-based recommendation system works."""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.loader import DataLoader, load_config
from src.models.graph_models import GraphBasedRecommender

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_system():
    """Test the basic functionality of the recommendation system."""
    logger.info("Testing Graph-Based Recommendation System")
    
    try:
        # Load configuration
        config = load_config("configs/config.yaml")
        logger.info("Configuration loaded successfully")
        
        # Load data
        data_loader = DataLoader(config)
        interactions_df, items_df, users_df = data_loader.load_data()
        logger.info(f"Data loaded: {len(interactions_df)} interactions, {len(items_df)} items, {len(users_df)} users")
        
        # Filter data
        interactions_df, items_df, users_df = data_loader.filter_data(
            interactions_df, items_df, users_df
        )
        logger.info(f"After filtering: {len(interactions_df)} interactions, {len(items_df)} items, {len(users_df)} users")
        
        # Create train/test splits
        train_df, val_df, test_df = data_loader.create_train_test_split(interactions_df)
        logger.info(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Initialize recommender
        recommender = GraphBasedRecommender(config)
        logger.info("Recommender initialized")
        
        # Test Random Walk model
        logger.info("Training Random Walk model...")
        recommender.train_model('random_walk', train_df, val_df)
        logger.info("Random Walk model trained successfully")
        
        # Test recommendations
        test_users = test_df['user_id'].unique()[:5]  # Test on first 5 users
        logger.info(f"Testing recommendations for {len(test_users)} users")
        
        for user_id in test_users:
            recommendations = recommender.recommend(user_id, 'random_walk', top_k=5)
            logger.info(f"User {user_id}: {len(recommendations)} recommendations")
            if recommendations:
                logger.info(f"  Sample recommendations: {recommendations[:3]}")
        
        logger.info("System test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"System test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_system()
    if success:
        print("\n✅ All tests passed! The graph-based recommendation system is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the logs for details.")
        sys.exit(1)
