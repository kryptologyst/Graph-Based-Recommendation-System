"""Graph-based recommendation models."""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging

# Optional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data, DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    GCNConv = None
    SAGEConv = None
    GATConv = None
    Data = None
    DataLoader = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cosine_similarity = None

try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False
    Node2Vec = None

logger = logging.getLogger(__name__)


class RandomWalkRecommender:
    """Random walk-based recommendation model."""
    
    def __init__(self, config: Dict):
        """Initialize random walk recommender.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.graph = None
        self.user_items = {}
        self.item_users = {}
        
    def build_graph(self, interactions_df: pd.DataFrame) -> None:
        """Build user-item interaction graph.
        
        Args:
            interactions_df: DataFrame with user-item interactions
        """
        logger.info("Building user-item interaction graph")
        
        self.graph = nx.Graph()
        self.user_items = {}
        self.item_users = {}
        
        # Add edges with weights
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            weight = row.get('rating', 1)
            
            # Add edge
            self.graph.add_edge(user_id, item_id, weight=weight)
            
            # Track user-item and item-user mappings
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)
            
            if item_id not in self.item_users:
                self.item_users[item_id] = set()
            self.item_users[item_id].add(user_id)
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def recommend(self, user_id: str, top_k: int = 10) -> List[str]:
        """Recommend items for a user using random walks.
        
        Args:
            user_id: User ID to recommend for
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended item IDs
        """
        if user_id not in self.user_items:
            return []
        
        num_walks = self.config['models']['random_walk']['num_walks']
        walk_length = self.config['models']['random_walk']['walk_length']
        
        recommended_items = []
        
        for _ in range(num_walks):
            current_node = user_id
            walk = []
            
            for _ in range(walk_length):
                if current_node not in self.graph:
                    break
                    
                neighbors = list(self.graph.neighbors(current_node))
                if not neighbors:
                    break
                
                # Weighted random choice based on edge weights
                weights = [self.graph[current_node][neighbor].get('weight', 1) for neighbor in neighbors]
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                next_node = np.random.choice(neighbors, p=weights)
                walk.append(next_node)
                current_node = next_node
            
            recommended_items.extend(walk)
        
        # Count recommendations and filter out already interacted items
        item_counts = pd.Series(recommended_items).value_counts()
        interacted_items = self.user_items[user_id]
        
        # Filter out items user has already interacted with
        item_counts = item_counts[~item_counts.index.isin(interacted_items)]
        
        # Only return item IDs (not user IDs)
        item_recommendations = []
        for item_id in item_counts.head(top_k).index.tolist():
            if item_id.startswith('item_'):
                item_recommendations.append(item_id)
        
        return item_recommendations


if TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
    class LightGCN(nn.Module):
        """LightGCN model for graph-based recommendation."""
        
        def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int):
            """Initialize LightGCN model.
            
            Args:
                num_users: Number of users
                num_items: Number of items
                embedding_dim: Embedding dimension
                num_layers: Number of GCN layers
            """
            super(LightGCN, self).__init__()
            
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            self.num_layers = num_layers
            
            # User and item embeddings
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            
            # Initialize embeddings
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
            
        def forward(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass of LightGCN.
            
            Args:
                edge_index: Edge index tensor
                
            Returns:
                Tuple of (user_embeddings, item_embeddings)
            """
            # Get initial embeddings
            user_embeddings = self.user_embedding.weight
            item_embeddings = self.item_embedding.weight
            
            # Stack embeddings
            all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
            
            # LightGCN: simple average aggregation
            embeddings_list = [all_embeddings]
            
            for _ in range(self.num_layers):
                all_embeddings = torch.sparse.mm(edge_index, all_embeddings)
                embeddings_list.append(all_embeddings)
            
            # Average all layer embeddings
            final_embeddings = torch.mean(torch.stack(embeddings_list), dim=0)
            
            # Split back to users and items
            user_embeddings = final_embeddings[:self.num_users]
            item_embeddings = final_embeddings[self.num_users:]
            
            return user_embeddings, item_embeddings
        
        def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                    edge_index: torch.Tensor) -> torch.Tensor:
            """Predict user-item scores.
            
            Args:
                user_ids: User ID tensor
                item_ids: Item ID tensor
                edge_index: Edge index tensor
                
            Returns:
                Predicted scores
            """
            user_embeddings, item_embeddings = self.forward(edge_index)
            
            user_emb = user_embeddings[user_ids]
            item_emb = item_embeddings[item_ids]
            
            scores = torch.sum(user_emb * item_emb, dim=1)
            return scores
else:
    # Fallback class when PyTorch/PyTorch Geometric is not available
    class LightGCN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Geometric are required for LightGCN model")


if TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
    class GraphSAGE(nn.Module):
        """GraphSAGE model for graph-based recommendation."""
        
        def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int):
            """Initialize GraphSAGE model.
            
            Args:
                num_users: Number of users
                num_items: Number of items
                embedding_dim: Embedding dimension
                num_layers: Number of SAGE layers
            """
            super(GraphSAGE, self).__init__()
            
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            
            # User and item embeddings
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            
            # GraphSAGE layers
            self.sage_layers = nn.ModuleList()
            for i in range(num_layers):
                self.sage_layers.append(SAGEConv(embedding_dim, embedding_dim))
            
            # Initialize embeddings
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
            
        def forward(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass of GraphSAGE.
            
            Args:
                edge_index: Edge index tensor
                
            Returns:
                Tuple of (user_embeddings, item_embeddings)
            """
            # Get initial embeddings
            user_embeddings = self.user_embedding.weight
            item_embeddings = self.item_embedding.weight
            
            # Stack embeddings
            all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
            
            # Apply GraphSAGE layers
            for sage_layer in self.sage_layers:
                all_embeddings = F.relu(sage_layer(all_embeddings, edge_index))
            
            # Split back to users and items
            user_embeddings = all_embeddings[:self.num_users]
            item_embeddings = all_embeddings[self.num_users:]
            
            return user_embeddings, item_embeddings
        
        def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                    edge_index: torch.Tensor) -> torch.Tensor:
            """Predict user-item scores.
            
            Args:
                user_ids: User ID tensor
                item_ids: Item ID tensor
                edge_index: Edge index tensor
                
            Returns:
                Predicted scores
            """
            user_embeddings, item_embeddings = self.forward(edge_index)
            
            user_emb = user_embeddings[user_ids]
            item_emb = item_embeddings[item_ids]
            
            scores = torch.sum(user_emb * item_emb, dim=1)
            return scores
else:
    # Fallback class when PyTorch/PyTorch Geometric is not available
    class GraphSAGE:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and PyTorch Geometric are required for GraphSAGE model")


if NODE2VEC_AVAILABLE and SKLEARN_AVAILABLE:
    class Node2VecRecommender:
        """Node2Vec-based recommendation model."""
        
        def __init__(self, config: Dict):
            """Initialize Node2Vec recommender.
            
            Args:
                config: Configuration dictionary
            """
            self.config = config
            self.model = None
            self.user_embeddings = None
            self.item_embeddings = None
            self.user_to_idx = {}
            self.item_to_idx = {}
            self.idx_to_user = {}
            self.idx_to_item = {}
            
        def build_graph(self, interactions_df: pd.DataFrame) -> None:
            """Build user-item interaction graph.
            
            Args:
                interactions_df: DataFrame with user-item interactions
            """
            logger.info("Building graph for Node2Vec")
            
            # Create user and item mappings
            users = sorted(interactions_df['user_id'].unique())
            items = sorted(interactions_df['item_id'].unique())
            
            self.user_to_idx = {user: idx for idx, user in enumerate(users)}
            self.item_to_idx = {item: idx for idx, item in enumerate(items)}
            self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
            self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for user in users:
                G.add_node(user, node_type='user')
            for item in items:
                G.add_node(item, node_type='item')
            
            # Add edges
            for _, row in interactions_df.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                weight = row.get('rating', 1)
                G.add_edge(user_id, item_id, weight=weight)
            
            # Train Node2Vec
            node2vec_config = self.config['models']['node2vec']
            node2vec = Node2Vec(
                G,
                dimensions=node2vec_config['embedding_dim'],
                walk_length=node2vec_config['walk_length'],
                num_walks=node2vec_config['num_walks'],
                p=node2vec_config['p'],
                q=node2vec_config['q'],
                workers=4
            )
            
            self.model = node2vec.fit(window=node2vec_config['window_size'], 
                                     min_count=1, batch_words=4)
            
            # Extract embeddings
            self.user_embeddings = {}
            self.item_embeddings = {}
            
            for user in users:
                if user in self.model.wv:
                    self.user_embeddings[user] = self.model.wv[user]
            
            for item in items:
                if item in self.model.wv:
                    self.item_embeddings[item] = self.model.wv[item]
            
            logger.info(f"Trained Node2Vec with {len(self.user_embeddings)} user embeddings and {len(self.item_embeddings)} item embeddings")
        
        def recommend(self, user_id: str, top_k: int = 10) -> List[str]:
            """Recommend items for a user using Node2Vec embeddings.
            
            Args:
                user_id: User ID to recommend for
                top_k: Number of recommendations to return
                
            Returns:
                List of recommended item IDs
            """
            if user_id not in self.user_embeddings:
                return []
            
            user_embedding = self.user_embeddings[user_id]
            
            # Compute similarities with all items
            similarities = []
            for item_id, item_embedding in self.item_embeddings.items():
                similarity = cosine_similarity([user_embedding], [item_embedding])[0][0]
                similarities.append((item_id, similarity))
            
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            recommendations = [item_id for item_id, _ in similarities[:top_k]]
            
            return recommendations
else:
    # Fallback class when Node2Vec/sklearn is not available
    class Node2VecRecommender:
        def __init__(self, *args, **kwargs):
            raise ImportError("Node2Vec and scikit-learn are required for Node2VecRecommender model")


class GraphBasedRecommender:
    """Main graph-based recommendation system."""
    
    def __init__(self, config: Dict):
        """Initialize graph-based recommender.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        
    def prepare_data(self, interactions_df: pd.DataFrame) -> Tuple[torch.Tensor, Dict]:
        """Prepare data for PyTorch Geometric models.
        
        Args:
            interactions_df: DataFrame with user-item interactions
            
        Returns:
            Tuple of (edge_index, mappings)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for prepare_data method")
            
        # Create mappings
        users = sorted(interactions_df['user_id'].unique())
        items = sorted(interactions_df['item_id'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.item_to_idx = {item: idx + len(users) for idx, item in enumerate(items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Create edge index
        edges = []
        for _, row in interactions_df.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['item_id']]
            edges.append([user_idx, item_idx])
            edges.append([item_idx, user_idx])  # Undirected graph
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        mappings = {
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item,
            'num_users': len(users),
            'num_items': len(items)
        }
        
        return edge_index, mappings
    
    def train_model(self, model_name: str, train_df: pd.DataFrame, 
                   val_df: pd.DataFrame) -> None:
        """Train a specific model.
        
        Args:
            model_name: Name of the model to train
            train_df: Training data
            val_df: Validation data
        """
        logger.info(f"Training {model_name} model")
        
        if model_name == 'random_walk':
            model = RandomWalkRecommender(self.config)
            model.build_graph(train_df)
            self.models[model_name] = model
            
        elif model_name == 'node2vec':
            model = Node2VecRecommender(self.config)
            model.build_graph(train_df)
            self.models[model_name] = model
            
        elif model_name in ['lightgcn', 'graphsage']:
            if not TORCH_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
                logger.warning(f"PyTorch/PyTorch Geometric not available, skipping {model_name}")
                return
                
            # Prepare data for PyTorch models
            edge_index, mappings = self.prepare_data(train_df)
            
            if model_name == 'lightgcn':
                model = LightGCN(
                    mappings['num_users'],
                    mappings['num_items'],
                    self.config['models']['lightgcn']['embedding_dim'],
                    self.config['models']['lightgcn']['num_layers']
                )
            else:  # graphsage
                model = GraphSAGE(
                    mappings['num_users'],
                    mappings['num_items'],
                    self.config['models']['graphsage']['embedding_dim'],
                    self.config['models']['graphsage']['num_layers']
                )
            
            # Train the model (simplified training loop)
            optimizer = torch.optim.Adam(model.parameters(), 
                                       lr=self.config['models'][model_name]['learning_rate'])
            
            model.train()
            for epoch in range(self.config['models'][model_name]['epochs']):
                optimizer.zero_grad()
                
                # Forward pass
                user_embeddings, item_embeddings = model(edge_index)
                
                # Simple loss: encourage user-item embeddings to be similar for positive pairs
                loss = 0
                for _, row in train_df.iterrows():
                    user_idx = mappings['user_to_idx'][row['user_id']]
                    item_idx = mappings['item_to_idx'][row['item_id']]
                    
                    user_emb = user_embeddings[user_idx]
                    item_emb = item_embeddings[item_idx - mappings['num_users']]
                    
                    # Cosine similarity loss
                    similarity = F.cosine_similarity(user_emb.unsqueeze(0), item_emb.unsqueeze(0))
                    loss += (1 - similarity) ** 2
                
                loss = loss / len(train_df)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Store model and mappings
            self.models[model_name] = model
            self.models[f"{model_name}_mappings"] = mappings
            self.models[f"{model_name}_edge_index"] = edge_index
    
    def recommend(self, user_id: str, model_name: str, top_k: int = 10) -> List[str]:
        """Get recommendations from a specific model.
        
        Args:
            user_id: User ID to recommend for
            model_name: Name of the model to use
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended item IDs
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return []
        
        model = self.models[model_name]
        
        if model_name in ['random_walk', 'node2vec']:
            return model.recommend(user_id, top_k)
        
        elif model_name in ['lightgcn', 'graphsage']:
            if not TORCH_AVAILABLE:
                logger.warning(f"PyTorch not available for {model_name}")
                return []
                
            mappings = self.models[f"{model_name}_mappings"]
            edge_index = self.models[f"{model_name}_edge_index"]
            
            if user_id not in mappings['user_to_idx']:
                return []
            
            user_idx = mappings['user_to_idx'][user_id]
            
            # Get user embedding
            model.eval()
            with torch.no_grad():
                user_embeddings, item_embeddings = model(edge_index)
                user_emb = user_embeddings[user_idx]
                
                # Compute similarities with all items
                similarities = torch.mm(user_emb.unsqueeze(0), item_embeddings.t()).squeeze(0)
                
                # Get top-k items
                _, top_indices = torch.topk(similarities, top_k)
                
                recommendations = []
                for idx in top_indices:
                    item_idx = idx.item() + mappings['num_users']
                    item_id = mappings['idx_to_item'][item_idx]
                    recommendations.append(item_id)
                
                return recommendations
        
        return []
