"""Streamlit demo for graph-based recommendation system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import yaml
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader, load_config
from src.models.graph_models import GraphBasedRecommender
from src.utils.evaluation import RecommendationEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Graph-Based Recommendation System",
    page_icon="🕸️",
    layout="wide"
)

# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'config' not in st.session_state:
    st.session_state.config = None


@st.cache_data
def load_data_and_models():
    """Load data and trained models."""
    try:
        # Load configuration
        config = load_config("configs/config.yaml")
        
        # Load data
        data_loader = DataLoader(config)
        interactions_df, items_df, users_df = data_loader.load_data()
        
        # Filter data
        interactions_df, items_df, users_df = data_loader.filter_data(
            interactions_df, items_df, users_df
        )
        
        # Create train/test splits
        train_df, val_df, test_df = data_loader.create_train_test_split(interactions_df)
        
        # Initialize recommender
        recommender = GraphBasedRecommender(config)
        
        # Load trained models (simplified - in practice, you'd load from saved checkpoints)
        models_to_load = ['random_walk', 'node2vec']
        
        for model_name in models_to_load:
            try:
                recommender.train_model(model_name, train_df, val_df)
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {str(e)}")
        
        return {
            'config': config,
            'recommender': recommender,
            'interactions_df': interactions_df,
            'items_df': items_df,
            'users_df': users_df,
            'train_df': train_df,
            'test_df': test_df
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def display_user_profile(user_id: str, data: Dict):
    """Display user profile information."""
    users_df = data['users_df']
    interactions_df = data['interactions_df']
    items_df = data['items_df']
    
    # Get user info
    user_info = users_df[users_df['user_id'] == user_id]
    
    if len(user_info) == 0:
        st.warning(f"User {user_id} not found in dataset")
        return
    
    user_info = user_info.iloc[0]
    
    # Get user interactions
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("User Information")
        st.write(f"**User ID:** {user_id}")
        st.write(f"**Age:** {user_info['age']}")
        st.write(f"**Gender:** {user_info['gender']}")
        st.write(f"**Location:** {user_info['location']}")
    
    with col2:
        st.subheader("Interaction Statistics")
        st.write(f"**Total Interactions:** {len(user_interactions)}")
        st.write(f"**Average Rating:** {user_interactions['rating'].mean():.2f}")
        st.write(f"**Rating Range:** {user_interactions['rating'].min()} - {user_interactions['rating'].max()}")
    
    with col3:
        st.subheader("Interaction History")
        if len(user_interactions) > 0:
            # Show recent interactions
            recent_interactions = user_interactions.nlargest(5, 'timestamp')
            for _, interaction in recent_interactions.iterrows():
                item_info = items_df[items_df['item_id'] == interaction['item_id']]
                if len(item_info) > 0:
                    item_title = item_info.iloc[0]['title']
                    st.write(f"• {item_title} (Rating: {interaction['rating']})")


def display_recommendations(user_id: str, model_name: str, data: Dict):
    """Display recommendations for a user."""
    recommender = data['recommender']
    items_df = data['items_df']
    
    # Get recommendations
    recommendations = recommender.recommend(user_id, model_name, top_k=10)
    
    if not recommendations:
        st.warning(f"No recommendations available for user {user_id} with model {model_name}")
        return
    
    st.subheader(f"Recommendations from {model_name.replace('_', ' ').title()}")
    
    # Display recommendations
    for i, item_id in enumerate(recommendations, 1):
        item_info = items_df[items_df['item_id'] == item_id]
        
        if len(item_info) > 0:
            item = item_info.iloc[0]
            
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.write(f"**#{i}**")
            
            with col2:
                st.write(f"**{item['title']}**")
                st.write(f"Category: {item['category']}")
                st.write(f"Tags: {item['tags']}")
                st.write(f"Price: ${item['price']:.2f}")
            
            with col3:
                st.write(f"**Item ID:** {item_id}")


def display_model_comparison(data: Dict):
    """Display model comparison metrics."""
    st.subheader("Model Performance Comparison")
    
    # This would typically load from saved results
    # For demo purposes, we'll create some sample metrics
    models = ['Random Walk', 'Node2Vec', 'LightGCN', 'GraphSAGE']
    
    metrics_data = {
        'Model': models,
        'Precision@5': [0.12, 0.15, 0.18, 0.16],
        'Recall@5': [0.08, 0.10, 0.12, 0.11],
        'NDCG@5': [0.14, 0.17, 0.20, 0.18],
        'Hit Rate@5': [0.25, 0.30, 0.35, 0.32],
        'Precision@10': [0.10, 0.12, 0.14, 0.13],
        'Recall@10': [0.15, 0.18, 0.22, 0.20],
        'NDCG@10': [0.18, 0.22, 0.26, 0.24],
        'Hit Rate@10': [0.40, 0.45, 0.50, 0.47]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.dataframe(metrics_df, use_container_width=True)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Precision@K', 'Recall@K', 'NDCG@K', 'Hit Rate@K'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    k_values = [5, 10]
    metrics = ['Precision', 'Recall', 'NDCG', 'Hit Rate']
    
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        
        for model in models:
            values = [metrics_df[metrics_df['Model'] == model][f'{metric}@{k}'].iloc[0] for k in k_values]
            fig.add_trace(
                go.Scatter(x=k_values, y=values, name=model, mode='lines+markers'),
                row=row, col=col
            )
    
    fig.update_layout(height=600, showlegend=True, title_text="Model Performance Metrics")
    st.plotly_chart(fig, use_container_width=True)


def display_data_overview(data: Dict):
    """Display data overview and statistics."""
    st.subheader("Dataset Overview")
    
    interactions_df = data['interactions_df']
    items_df = data['items_df']
    users_df = data['users_df']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(users_df))
    
    with col2:
        st.metric("Total Items", len(items_df))
    
    with col3:
        st.metric("Total Interactions", len(interactions_df))
    
    with col4:
        sparsity = 1 - (len(interactions_df) / (len(users_df) * len(items_df)))
        st.metric("Sparsity", f"{sparsity:.3f}")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    rating_counts = interactions_df['rating'].value_counts().sort_index()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title="Distribution of Ratings",
        labels={'x': 'Rating', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Category distribution
    st.subheader("Item Categories")
    category_counts = items_df['category'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Distribution of Item Categories"
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit app."""
    st.title("🕸️ Graph-Based Recommendation System")
    st.markdown("Explore graph-based recommendation algorithms and their performance")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        data = load_data_and_models()
    
    if data is None:
        st.error("Failed to load data. Please check the configuration and data files.")
        return
    
    # Store in session state
    st.session_state.data = data
    st.session_state.recommender = data['recommender']
    st.session_state.config = data['config']
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Overview", "User Recommendations", "Model Comparison", "Item Similarity"]
    )
    
    if page == "Data Overview":
        display_data_overview(data)
    
    elif page == "User Recommendations":
        st.header("User Recommendations")
        
        # User selection
        users = data['users_df']['user_id'].tolist()
        selected_user = st.selectbox("Select a user", users)
        
        if selected_user:
            # Display user profile
            display_user_profile(selected_user, data)
            
            st.divider()
            
            # Model selection
            available_models = ['random_walk', 'node2vec']
            selected_model = st.selectbox("Select a model", available_models)
            
            if selected_model:
                display_recommendations(selected_user, selected_model, data)
    
    elif page == "Model Comparison":
        display_model_comparison(data)
    
    elif page == "Item Similarity":
        st.header("Item Similarity Search")
        
        # Item selection
        items = data['items_df']['item_id'].tolist()
        selected_item = st.selectbox("Select an item", items)
        
        if selected_item:
            # Display item info
            item_info = data['items_df'][data['items_df']['item_id'] == selected_item].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Item Information")
                st.write(f"**Title:** {item_info['title']}")
                st.write(f"**Category:** {item_info['category']}")
                st.write(f"**Tags:** {item_info['tags']}")
                st.write(f"**Price:** ${item_info['price']:.2f}")
            
            with col2:
                st.subheader("Similar Items")
                # This would typically use item embeddings for similarity
                # For demo, we'll show items from the same category
                similar_items = data['items_df'][
                    (data['items_df']['category'] == item_info['category']) &
                    (data['items_df']['item_id'] != selected_item)
                ].head(5)
                
                for _, item in similar_items.iterrows():
                    st.write(f"• {item['title']} (ID: {item['item_id']})")


if __name__ == "__main__":
    main()
