"""Unit tests for graph-based recommendation system."""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataGenerator, DataLoader
from src.models.graph_models import RandomWalkRecommender, Node2VecRecommender, GraphBasedRecommender
from src.utils.evaluation import RecommendationEvaluator, CoverageEvaluator


class TestDataGenerator:
    """Test data generation functionality."""
    
    def test_init(self):
        """Test DataGenerator initialization."""
        config = {'random_seed': 42}
        generator = DataGenerator(config)
        assert generator.config == config
        assert generator.random_state is not None
    
    def test_generate_interactions(self):
        """Test interaction generation."""
        config = {'random_seed': 42}
        generator = DataGenerator(config)
        
        interactions_df = generator.generate_interactions(n_users=10, n_items=5)
        
        assert len(interactions_df) > 0
        assert 'user_id' in interactions_df.columns
        assert 'item_id' in interactions_df.columns
        assert 'rating' in interactions_df.columns
        assert 'timestamp' in interactions_df.columns
        
        # Check rating range
        assert interactions_df['rating'].min() >= 1
        assert interactions_df['rating'].max() <= 5
    
    def test_generate_items(self):
        """Test item generation."""
        config = {'random_seed': 42}
        generator = DataGenerator(config)
        
        items_df = generator.generate_items(n_items=10)
        
        assert len(items_df) == 10
        assert 'item_id' in items_df.columns
        assert 'title' in items_df.columns
        assert 'category' in items_df.columns
        assert 'tags' in items_df.columns
        assert 'price' in items_df.columns
    
    def test_generate_users(self):
        """Test user generation."""
        config = {'random_seed': 42}
        generator = DataGenerator(config)
        
        users_df = generator.generate_users(n_users=10)
        
        assert len(users_df) == 10
        assert 'user_id' in users_df.columns
        assert 'age' in users_df.columns
        assert 'gender' in users_df.columns
        assert 'location' in users_df.columns


class TestRandomWalkRecommender:
    """Test RandomWalkRecommender functionality."""
    
    def test_init(self):
        """Test RandomWalkRecommender initialization."""
        config = {'models': {'random_walk': {'num_walks': 10, 'walk_length': 5}}}
        recommender = RandomWalkRecommender(config)
        assert recommender.config == config
        assert recommender.graph is None
    
    def test_build_graph(self):
        """Test graph building."""
        config = {'models': {'random_walk': {'num_walks': 10, 'walk_length': 5}}}
        recommender = RandomWalkRecommender(config)
        
        # Create sample interactions
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2'],
            'item_id': ['item1', 'item2', 'item1'],
            'rating': [5, 4, 3]
        })
        
        recommender.build_graph(interactions_df)
        
        assert recommender.graph is not None
        assert len(recommender.user_items) == 2
        assert len(recommender.item_users) == 2
    
    def test_recommend(self):
        """Test recommendation generation."""
        config = {'models': {'random_walk': {'num_walks': 10, 'walk_length': 5}}}
        recommender = RandomWalkRecommender(config)
        
        # Create sample interactions
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'item_id': ['item1', 'item2', 'item1', 'item3'],
            'rating': [5, 4, 3, 2]
        })
        
        recommender.build_graph(interactions_df)
        
        # Test recommendation for existing user
        recommendations = recommender.recommend('user1', top_k=3)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        # Test recommendation for non-existing user
        recommendations = recommender.recommend('nonexistent_user', top_k=3)
        assert recommendations == []


class TestRecommendationEvaluator:
    """Test RecommendationEvaluator functionality."""
    
    def test_init(self):
        """Test RecommendationEvaluator initialization."""
        config = {
            'evaluation': {
                'metrics': ['precision@5', 'recall@5'],
                'k_values': [5, 10]
            }
        }
        evaluator = RecommendationEvaluator(config)
        assert evaluator.config == config
        assert evaluator.metrics == ['precision@5', 'recall@5']
        assert evaluator.k_values == [5, 10]
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        config = {'evaluation': {'metrics': [], 'k_values': []}}
        evaluator = RecommendationEvaluator(config)
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        precision = evaluator.precision_at_k(recommendations, relevant_items, k=5)
        expected = 2 / 5  # 2 relevant items out of 5 recommendations
        assert precision == expected
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        config = {'evaluation': {'metrics': [], 'k_values': []}}
        evaluator = RecommendationEvaluator(config)
        
        recommendations = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant_items = ['item1', 'item3', 'item6']
        
        recall = evaluator.recall_at_k(recommendations, relevant_items, k=5)
        expected = 2 / 3  # 2 relevant items found out of 3 total relevant
        assert recall == expected
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        config = {'evaluation': {'metrics': [], 'k_values': []}}
        evaluator = RecommendationEvaluator(config)
        
        recommendations = ['item1', 'item2', 'item3']
        relevant_items = ['item1', 'item3']
        
        ndcg = evaluator.ndcg_at_k(recommendations, relevant_items, k=3)
        assert 0 <= ndcg <= 1
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        config = {'evaluation': {'metrics': [], 'k_values': []}}
        evaluator = RecommendationEvaluator(config)
        
        recommendations = ['item1', 'item2', 'item3']
        relevant_items = ['item1', 'item4']
        
        hit_rate = evaluator.hit_rate_at_k(recommendations, relevant_items, k=3)
        assert hit_rate == 1.0  # item1 is in recommendations
        
        recommendations = ['item2', 'item3', 'item5']
        hit_rate = evaluator.hit_rate_at_k(recommendations, relevant_items, k=3)
        assert hit_rate == 0.0  # no relevant items in recommendations


class TestCoverageEvaluator:
    """Test CoverageEvaluator functionality."""
    
    def test_catalog_coverage(self):
        """Test catalog coverage calculation."""
        evaluator = CoverageEvaluator()
        
        model_recommendations = {
            'user1': ['item1', 'item2'],
            'user2': ['item2', 'item3'],
            'user3': ['item1', 'item4']
        }
        total_items = 10
        
        coverage = evaluator.catalog_coverage(model_recommendations, total_items)
        expected = 4 / 10  # 4 unique items out of 10 total
        assert coverage == expected
    
    def test_user_coverage(self):
        """Test user coverage calculation."""
        evaluator = CoverageEvaluator()
        
        model_recommendations = {
            'user1': ['item1', 'item2'],
            'user2': ['item2', 'item3']
        }
        total_users = 5
        
        coverage = evaluator.user_coverage(model_recommendations, total_users)
        expected = 2 / 5  # 2 users with recommendations out of 5 total
        assert coverage == expected
    
    def test_intra_list_diversity(self):
        """Test intra-list diversity calculation."""
        evaluator = CoverageEvaluator()
        
        recommendations = ['item1', 'item2', 'item3', 'item1', 'item2']
        diversity = evaluator.intra_list_diversity(recommendations)
        expected = 3 / 5  # 3 unique items out of 5 total
        assert diversity == expected


class TestGraphBasedRecommender:
    """Test GraphBasedRecommender functionality."""
    
    def test_init(self):
        """Test GraphBasedRecommender initialization."""
        config = {'models': {}}
        recommender = GraphBasedRecommender(config)
        assert recommender.config == config
        assert recommender.models == {}
    
    def test_prepare_data(self):
        """Test data preparation for PyTorch models."""
        config = {'models': {}}
        recommender = GraphBasedRecommender(config)
        
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2'],
            'item_id': ['item1', 'item2', 'item1'],
            'rating': [5, 4, 3]
        })
        
        edge_index, mappings = recommender.prepare_data(interactions_df)
        
        assert isinstance(edge_index, torch.Tensor)
        assert 'user_to_idx' in mappings
        assert 'item_to_idx' in mappings
        assert 'num_users' in mappings
        assert 'num_items' in mappings


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_random_walk(self):
        """Test end-to-end Random Walk recommendation."""
        # Generate sample data
        config = {'random_seed': 42, 'models': {'random_walk': {'num_walks': 10, 'walk_length': 5}}}
        generator = DataGenerator(config)
        
        interactions_df = generator.generate_interactions(n_users=20, n_items=10)
        
        # Train Random Walk model
        recommender = RandomWalkRecommender(config)
        recommender.build_graph(interactions_df)
        
        # Get recommendations
        users = interactions_df['user_id'].unique()
        test_user = users[0]
        recommendations = recommender.recommend(test_user, top_k=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
    
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline."""
        config = {
            'evaluation': {
                'metrics': ['precision@5', 'recall@5'],
                'k_values': [5]
            }
        }
        
        # Create sample data
        model_recommendations = {
            'user1': ['item1', 'item2', 'item3', 'item4', 'item5'],
            'user2': ['item2', 'item3', 'item4', 'item5', 'item6']
        }
        
        test_interactions = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'item_id': ['item1', 'item3', 'item2', 'item4'],
            'rating': [5, 4, 3, 2]
        })
        
        evaluator = RecommendationEvaluator(config)
        metrics = evaluator.evaluate_model(model_recommendations, test_interactions)
        
        assert 'precision@5' in metrics
        assert 'recall@5' in metrics
        assert all(0 <= score <= 1 for score in metrics.values())


if __name__ == "__main__":
    pytest.main([__file__])
