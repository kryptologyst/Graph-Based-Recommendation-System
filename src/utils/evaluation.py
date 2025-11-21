"""Evaluation metrics and utilities for recommendation systems."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Evaluator for recommendation system metrics."""
    
    def __init__(self, config: Dict):
        """Initialize evaluator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics = config['evaluation']['metrics']
        self.k_values = config['evaluation']['k_values']
    
    def precision_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        if len(top_k_recs) == 0:
            return 0.0
        
        precision = len([item for item in top_k_recs if item in relevant_set]) / len(top_k_recs)
        return precision
    
    def recall_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Recall@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        recall = len([item for item in top_k_recs if item in relevant_set]) / len(relevant_items)
        return recall
    
    def ndcg_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate NDCG@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        if len(relevant_set) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score (1 if hit, 0 otherwise)
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        hit = any(item in relevant_set for item in top_k_recs)
        return 1.0 if hit else 0.0
    
    def map_at_k(self, recommendations: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate MAP@K.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top recommendations to consider
            
        Returns:
            MAP@K score
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_set = set(relevant_items)
        
        if len(relevant_set) == 0:
            return 0.0
        
        # Calculate average precision
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items)
    
    def calculate_metrics(self, 
                         recommendations: List[str], 
                         relevant_items: List[str]) -> Dict[str, float]:
        """Calculate all metrics for given recommendations.
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: List of relevant item IDs
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        for k in self.k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(recommendations, relevant_items, k)
            metrics[f'recall@{k}'] = self.recall_at_k(recommendations, relevant_items, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(recommendations, relevant_items, k)
            metrics[f'hit_rate@{k}'] = self.hit_rate_at_k(recommendations, relevant_items, k)
            metrics[f'map@{k}'] = self.map_at_k(recommendations, relevant_items, k)
        
        return metrics
    
    def evaluate_model(self, 
                      model_recommendations: Dict[str, List[str]], 
                      test_interactions: pd.DataFrame) -> Dict[str, float]:
        """Evaluate a model on test data.
        
        Args:
            model_recommendations: Dictionary mapping user_id to list of recommended items
            test_interactions: Test interactions DataFrame
            
        Returns:
            Dictionary of average metric scores
        """
        logger.info("Evaluating model on test data")
        
        # Group test interactions by user
        user_relevant_items = defaultdict(list)
        for _, row in test_interactions.iterrows():
            user_relevant_items[row['user_id']].append(row['item_id'])
        
        # Calculate metrics for each user
        all_metrics = defaultdict(list)
        
        for user_id, recommendations in model_recommendations.items():
            if user_id not in user_relevant_items:
                continue
            
            relevant_items = user_relevant_items[user_id]
            user_metrics = self.calculate_metrics(recommendations, relevant_items)
            
            for metric, score in user_metrics.items():
                all_metrics[metric].append(score)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric, scores in all_metrics.items():
            avg_metrics[metric] = np.mean(scores)
        
        logger.info(f"Evaluated {len(model_recommendations)} users")
        return avg_metrics
    
    def evaluate_multiple_models(self, 
                               model_results: Dict[str, Dict[str, List[str]]], 
                               test_interactions: pd.DataFrame) -> pd.DataFrame:
        """Evaluate multiple models and create comparison table.
        
        Args:
            model_results: Dictionary mapping model_name to user recommendations
            test_interactions: Test interactions DataFrame
            
        Returns:
            DataFrame with model comparison results
        """
        logger.info("Evaluating multiple models")
        
        results = []
        
        for model_name, recommendations in model_results.items():
            logger.info(f"Evaluating {model_name}")
            metrics = self.evaluate_model(recommendations, test_interactions)
            
            result_row = {'model': model_name}
            result_row.update(metrics)
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        
        # Sort by NDCG@10 (or another primary metric)
        primary_metric = 'ndcg@10'
        if primary_metric in results_df.columns:
            results_df = results_df.sort_values(primary_metric, ascending=False)
        
        logger.info("Model evaluation completed")
        return results_df


class CoverageEvaluator:
    """Evaluator for coverage and diversity metrics."""
    
    def __init__(self):
        """Initialize coverage evaluator."""
        pass
    
    def catalog_coverage(self, 
                       model_recommendations: Dict[str, List[str]], 
                       total_items: int) -> float:
        """Calculate catalog coverage.
        
        Args:
            model_recommendations: Dictionary mapping user_id to recommendations
            total_items: Total number of items in catalog
            
        Returns:
            Catalog coverage score
        """
        recommended_items = set()
        
        for recommendations in model_recommendations.values():
            recommended_items.update(recommendations)
        
        coverage = len(recommended_items) / total_items
        return coverage
    
    def user_coverage(self, 
                     model_recommendations: Dict[str, List[str]], 
                     total_users: int) -> float:
        """Calculate user coverage.
        
        Args:
            model_recommendations: Dictionary mapping user_id to recommendations
            total_users: Total number of users
            
        Returns:
            User coverage score
        """
        users_with_recs = len(model_recommendations)
        coverage = users_with_recs / total_users
        return coverage
    
    def intra_list_diversity(self, 
                           recommendations: List[str], 
                           item_features: Optional[pd.DataFrame] = None) -> float:
        """Calculate intra-list diversity.
        
        Args:
            recommendations: List of recommended items
            item_features: DataFrame with item features (optional)
            
        Returns:
            Intra-list diversity score
        """
        if len(recommendations) <= 1:
            return 0.0
        
        if item_features is not None:
            # Use item features for diversity calculation
            # This is a simplified version - in practice, you'd use more sophisticated methods
            categories = []
            for item in recommendations:
                if item in item_features['item_id'].values:
                    category = item_features[item_features['item_id'] == item]['category'].iloc[0]
                    categories.append(category)
            
            if len(categories) == 0:
                return 0.0
            
            # Calculate diversity as 1 - (most common category proportion)
            category_counts = pd.Series(categories).value_counts()
            max_proportion = category_counts.iloc[0] / len(categories)
            diversity = 1 - max_proportion
            
            return diversity
        else:
            # Simple diversity based on unique items
            unique_items = len(set(recommendations))
            total_items = len(recommendations)
            diversity = unique_items / total_items
            
            return diversity
    
    def popularity_bias(self, 
                       model_recommendations: Dict[str, List[str]], 
                       item_popularity: Dict[str, int]) -> float:
        """Calculate popularity bias.
        
        Args:
            model_recommendations: Dictionary mapping user_id to recommendations
            item_popularity: Dictionary mapping item_id to popularity count
            
        Returns:
            Popularity bias score
        """
        if not item_popularity:
            return 0.0
        
        all_recommendations = []
        for recommendations in model_recommendations.values():
            all_recommendations.extend(recommendations)
        
        if not all_recommendations:
            return 0.0
        
        # Calculate average popularity of recommended items
        rec_popularities = []
        for item in all_recommendations:
            if item in item_popularity:
                rec_popularities.append(item_popularity[item])
        
        if not rec_popularities:
            return 0.0
        
        avg_rec_popularity = np.mean(rec_popularities)
        avg_overall_popularity = np.mean(list(item_popularity.values()))
        
        bias = avg_rec_popularity / avg_overall_popularity
        return bias


def create_evaluation_report(model_results: Dict[str, Dict[str, List[str]]], 
                           test_interactions: pd.DataFrame,
                           items_df: pd.DataFrame,
                           config: Dict) -> Dict:
    """Create comprehensive evaluation report.
    
    Args:
        model_results: Dictionary mapping model_name to user recommendations
        test_interactions: Test interactions DataFrame
        items_df: Items metadata DataFrame
        config: Configuration dictionary
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Creating comprehensive evaluation report")
    
    # Initialize evaluators
    rec_evaluator = RecommendationEvaluator(config)
    coverage_evaluator = CoverageEvaluator()
    
    # Calculate recommendation metrics
    rec_metrics_df = rec_evaluator.evaluate_multiple_models(model_results, test_interactions)
    
    # Calculate coverage metrics
    coverage_results = {}
    total_items = len(items_df)
    total_users = len(test_interactions['user_id'].unique())
    
    # Calculate item popularity
    item_popularity = test_interactions['item_id'].value_counts().to_dict()
    
    for model_name, recommendations in model_results.items():
        catalog_cov = coverage_evaluator.catalog_coverage(recommendations, total_items)
        user_cov = coverage_evaluator.user_coverage(recommendations, total_users)
        
        # Calculate average intra-list diversity
        diversities = []
        for user_recs in recommendations.values():
            diversity = coverage_evaluator.intra_list_diversity(user_recs, items_df)
            diversities.append(diversity)
        
        avg_diversity = np.mean(diversities) if diversities else 0.0
        
        # Calculate popularity bias
        bias = coverage_evaluator.popularity_bias(recommendations, item_popularity)
        
        coverage_results[model_name] = {
            'catalog_coverage': catalog_cov,
            'user_coverage': user_cov,
            'avg_intra_list_diversity': avg_diversity,
            'popularity_bias': bias
        }
    
    # Combine results
    report = {
        'recommendation_metrics': rec_metrics_df,
        'coverage_metrics': coverage_results,
        'summary': {
            'num_models': len(model_results),
            'num_users': total_users,
            'num_items': total_items,
            'num_test_interactions': len(test_interactions)
        }
    }
    
    logger.info("Evaluation report created")
    return report
