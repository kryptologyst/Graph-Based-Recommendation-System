"""Main training script for graph-based recommendation system."""

import os
import sys
import logging
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader, load_config
from src.models.graph_models import GraphBasedRecommender
from src.utils.evaluation import create_evaluation_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_random_seeds(config: Dict) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        config: Configuration dictionary
    """
    random_seed = config.get('random_seed', 42)
    numpy_seed = config.get('numpy_seed', 42)
    torch_seed = config.get('torch_seed', 42)
    
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
    
    logger.info(f"Set random seeds - random: {random_seed}, numpy: {numpy_seed}, torch: {torch_seed}")


def train_and_evaluate_models(config: Dict) -> Dict:
    """Train and evaluate all graph-based models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Starting model training and evaluation")
    
    # Load data
    data_loader = DataLoader(config)
    interactions_df, items_df, users_df = data_loader.load_data()
    
    # Filter data
    interactions_df, items_df, users_df = data_loader.filter_data(
        interactions_df, items_df, users_df
    )
    
    # Create train/test splits
    train_df, val_df, test_df = data_loader.create_train_test_split(interactions_df)
    
    logger.info(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize recommender
    recommender = GraphBasedRecommender(config)
    
    # Models to train
    models_to_train = ['random_walk', 'node2vec', 'lightgcn', 'graphsage']
    
    # Train models
    for model_name in models_to_train:
        logger.info(f"Training {model_name}")
        try:
            recommender.train_model(model_name, train_df, val_df)
            logger.info(f"Successfully trained {model_name}")
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {str(e)}")
            continue
    
    # Generate recommendations for all users in test set
    test_users = test_df['user_id'].unique()
    model_results = {}
    
    for model_name in models_to_train:
        if model_name in recommender.models:
            logger.info(f"Generating recommendations with {model_name}")
            recommendations = {}
            
            for user_id in test_users:
                try:
                    user_recs = recommender.recommend(user_id, model_name, top_k=20)
                    recommendations[user_id] = user_recs
                except Exception as e:
                    logger.warning(f"Failed to generate recommendations for user {user_id} with {model_name}: {str(e)}")
                    recommendations[user_id] = []
            
            model_results[model_name] = recommendations
            logger.info(f"Generated recommendations for {len(recommendations)} users with {model_name}")
    
    # Evaluate models
    logger.info("Evaluating models")
    evaluation_results = create_evaluation_report(
        model_results, test_df, items_df, config
    )
    
    return evaluation_results


def save_results(results: Dict, output_dir: str = "results") -> None:
    """Save evaluation results to files.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Output directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save recommendation metrics
    rec_metrics_df = results['recommendation_metrics']
    rec_metrics_df.to_csv(output_path / "recommendation_metrics.csv", index=False)
    
    # Save coverage metrics
    coverage_df = pd.DataFrame(results['coverage_metrics']).T
    coverage_df.to_csv(output_path / "coverage_metrics.csv")
    
    # Save summary
    summary_df = pd.DataFrame([results['summary']])
    summary_df.to_csv(output_path / "evaluation_summary.csv", index=False)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """Main training function."""
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Load configuration
    config_path = "configs/config.yaml"
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    # Set random seeds
    set_random_seeds(config)
    
    # Train and evaluate models
    try:
        results = train_and_evaluate_models(config)
        
        # Save results
        save_results(results)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        print("\nRecommendation Metrics:")
        print(results['recommendation_metrics'].to_string(index=False))
        
        print("\nCoverage Metrics:")
        coverage_df = pd.DataFrame(results['coverage_metrics']).T
        print(coverage_df.to_string())
        
        print(f"\nSummary:")
        for key, value in results['summary'].items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
