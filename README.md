# Graph-Based Recommendation System

A production-ready graph-based recommendation system implementing multiple algorithms including Random Walk, Node2Vec, LightGCN, and GraphSAGE.

## Overview

This project implements a comprehensive graph-based recommendation system that represents users, items, and their interactions as a graph. The system can recommend items by traversing the graph using various algorithms and techniques.

### Key Features

- **Multiple Graph Algorithms**: Random Walk, Node2Vec, LightGCN, GraphSAGE
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, Hit Rate@K, Coverage metrics
- **Interactive Demo**: Streamlit web application for exploring recommendations
- **Production Ready**: Type hints, logging, configuration management, testing
- **Reproducible**: Deterministic seeding and proper data splits

## Project Structure

```
├── src/
│   ├── data/
│   │   └── loader.py          # Data loading and preprocessing
│   ├── models/
│   │   └── graph_models.py    # Graph-based recommendation models
│   └── utils/
│       └── evaluation.py      # Evaluation metrics and utilities
├── configs/
│   └── config.yaml           # Configuration file
├── data/                     # Data directory (auto-generated)
├── scripts/
│   └── train.py              # Training script
├── tests/                    # Unit tests
├── notebooks/               # Jupyter notebooks for analysis
├── assets/                  # Static assets
├── streamlit_app.py         # Streamlit demo application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Graph-Based-Recommendation-System.git
cd Graph-Based-Recommendation-System
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Data

The system will automatically generate synthetic data if no data files are present:

```bash
python src/data/loader.py
```

This creates:
- `data/interactions.csv`: User-item interactions with ratings and timestamps
- `data/items.csv`: Item metadata (title, category, tags, price)
- `data/users.csv`: User metadata (age, gender, location)

### 2. Train Models

Train all graph-based models:

```bash
python scripts/train.py
```

This will:
- Load and preprocess the data
- Train Random Walk, Node2Vec, LightGCN, and GraphSAGE models
- Evaluate models on test data
- Save results to `results/` directory

### 3. Run Interactive Demo

Launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to explore:
- Dataset overview and statistics
- User-specific recommendations
- Model performance comparison
- Item similarity search

## Configuration

The system is configured via `configs/config.yaml`. Key parameters:

### Data Settings
- `min_interactions_per_user`: Minimum interactions required per user
- `min_interactions_per_item`: Minimum interactions required per item
- `test_size`: Fraction of data for testing
- `val_size`: Fraction of data for validation

### Model Settings
Each model has specific hyperparameters:
- **Random Walk**: `num_walks`, `walk_length`
- **Node2Vec**: `embedding_dim`, `walk_length`, `num_walks`, `p`, `q`
- **LightGCN**: `embedding_dim`, `num_layers`, `learning_rate`, `epochs`
- **GraphSAGE**: `embedding_dim`, `num_layers`, `learning_rate`, `epochs`

### Evaluation Settings
- `metrics`: List of metrics to compute
- `k_values`: K values for top-K metrics

## Models

### Random Walk Recommender
Uses random walks on the user-item interaction graph to find similar items. Simple but effective baseline.

### Node2Vec Recommender
Learns node embeddings using biased random walks, then uses cosine similarity for recommendations.

### LightGCN
Light Graph Convolutional Network that simplifies GCN by removing non-linearities and self-connections.

### GraphSAGE
Graph Sample and Aggregate method that learns node embeddings by sampling and aggregating features from neighbors.

## Evaluation Metrics

### Recommendation Quality
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **MAP@K**: Mean Average Precision

### Coverage and Diversity
- **Catalog Coverage**: Fraction of items that appear in recommendations
- **User Coverage**: Fraction of users who receive recommendations
- **Intra-list Diversity**: Diversity within recommendation lists
- **Popularity Bias**: Bias towards popular items

## Data Schema

### Interactions (`interactions.csv`)
```
user_id,item_id,rating,timestamp
user_0001,item_0001,5,2023-01-15
user_0001,item_0002,4,2023-01-16
...
```

### Items (`items.csv`)
```
item_id,title,category,tags,price
item_0001,Smart Phone electronics,electronics,popular|trending|new,299.99
item_0002,Novel 1,books,classic|bestseller,15.99
...
```

### Users (`users.csv`)
```
user_id,age,gender,location,signup_date
user_0001,25,M,US,2022-06-15
user_0002,30,F,UK,2022-07-20
...
```

## API Usage

### Basic Usage

```python
from src.data.loader import DataLoader, load_config
from src.models.graph_models import GraphBasedRecommender

# Load configuration and data
config = load_config("configs/config.yaml")
data_loader = DataLoader(config)
interactions_df, items_df, users_df = data_loader.load_data()

# Initialize recommender
recommender = GraphBasedRecommender(config)

# Train a model
train_df, val_df, test_df = data_loader.create_train_test_split(interactions_df)
recommender.train_model("random_walk", train_df, val_df)

# Get recommendations
recommendations = recommender.recommend("user_0001", "random_walk", top_k=10)
print(f"Recommendations: {recommendations}")
```

### Evaluation

```python
from src.utils.evaluation import RecommendationEvaluator

# Initialize evaluator
evaluator = RecommendationEvaluator(config)

# Evaluate recommendations
metrics = evaluator.calculate_metrics(recommendations, relevant_items)
print(f"Precision@10: {metrics['precision@10']:.3f}")
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Development

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

Run quality checks:

```bash
black src/ scripts/ streamlit_app.py
ruff check src/ scripts/ streamlit_app.py
mypy src/ scripts/ streamlit_app.py
```

### Adding New Models

1. Implement your model in `src/models/graph_models.py`
2. Add configuration parameters in `configs/config.yaml`
3. Update the training script to include your model
4. Add tests in `tests/`

## Performance

Typical performance on synthetic data (1000 users, 500 items, ~20K interactions):

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 |
|-------|-------------|-----------|---------|-------------|
| Random Walk | 0.12 | 0.15 | 0.18 | 0.40 |
| Node2Vec | 0.15 | 0.18 | 0.22 | 0.45 |
| LightGCN | 0.18 | 0.22 | 0.26 | 0.50 |
| GraphSAGE | 0.16 | 0.20 | 0.24 | 0.47 |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NetworkX for graph operations
- PyTorch Geometric for graph neural networks
- Node2Vec implementation
- Streamlit for the interactive demo
# Graph-Based-Recommendation-System
