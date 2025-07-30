"""
Module: feature_engineering
Description: Create and configure feature transformation pipeline for earthquake data.
Enhanced with parameterized clustering, better feature naming, and leakage prevention.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class GeographicClusterer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for geographic clustering to avoid data leakage.
    Fits only on training data when used in a pipeline.
    """
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        
    def fit(self, X, y=None):
        """Fit KMeans on geographic coordinates"""
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        # X should be a 2D array with [latitude, longitude]
        self.kmeans.fit(X)
        return self
        
    def transform(self, X):
        """Transform coordinates to cluster labels"""
        if self.kmeans is None:
            raise ValueError("GeographicClusterer must be fitted before transform")
        cluster_labels = self.kmeans.predict(X)
        return cluster_labels.reshape(-1, 1)


def build_feature_pipeline(
    n_clusters: int = 10,
    depth_bins: int = 5,
    depth_strategy: str = 'quantile',
    include_temporal: bool = True,
    available_columns: list = None
) -> Pipeline:
    """
    Build a comprehensive preprocessing pipeline for earthquake data.
    
    Features created:
    - Geographic clustering to capture regional seismic patterns
    - Depth discretization (shallow/intermediate/deep zones)
    - Temporal features (if available)
    - Scaling of all numeric features
    - One-hot encoding of categorical features
    
    Args:
        n_clusters (int): Number of geographic clusters to create
        depth_bins (int): Number of depth bins to create
        depth_strategy (str): Strategy for depth binning ('uniform', 'quantile', 'kmeans')
        include_temporal (bool): Whether to include temporal features
        available_columns (list): List of available columns in the dataset
        
    Returns:
        Pipeline: Complete preprocessing pipeline
    """
    # Define feature groups based on available columns
    # Core geographic features (always present)
    geo_features = ['latitude', 'longitude']
    
    # Numeric features that need scaling
    numeric_features = ['latitude', 'longitude', 'depth']
    
    # Optional temporal features (if created during preprocessing)
    temporal_features = []
    if include_temporal:
        temporal_features = ['year', 'month', 'day', 'day_of_week', 'hour', 'is_weekend']
        numeric_features.extend(temporal_features)
    
    # Quality metrics - only include if available
    potential_quality_features = ['nst', 'gap', 'dmin', 'rms', 'magNst']
    quality_features = []
    if available_columns:
        quality_features = [f for f in potential_quality_features if f in available_columns]
        numeric_features.extend(quality_features)
    
    # Error measurements - only include if available  
    potential_error_features = ['horizontalError', 'depthError', 'magError']
    error_features = []
    if available_columns:
        error_features = [f for f in potential_error_features if f in available_columns]
        numeric_features.extend(error_features)
    
    # Categorical features - only include if available
    potential_categorical = ['magType', 'net', 'type', 'status', 'locationSource', 'magSource']
    categorical_features = []
    if available_columns:
        categorical_features = [f for f in potential_categorical if f in available_columns]
    
    logger.info(f"Building pipeline with {n_clusters} geographic clusters, {depth_bins} depth bins")
    
    # Create transformers
    transformers = []
    
    # 1. Geographic clustering (prevents leakage by fitting only on training data)
    geo_clusterer = GeographicClusterer(n_clusters=n_clusters, random_state=42)
    transformers.append(('geo_cluster', geo_clusterer, geo_features))
    
    # 2. Depth binning
    depth_discretizer = KBinsDiscretizer(
        n_bins=depth_bins, 
        encode='ordinal',  # Use ordinal to preserve depth ordering
        strategy=depth_strategy,
        random_state=42,
        dtype=np.float64
    )
    transformers.append(('depth_bin', depth_discretizer, ['depth']))
    
    # 3. Scale all numeric features
    numeric_scaler = StandardScaler()
    transformers.append(('scaler', numeric_scaler, numeric_features))
    
    # 4. One-hot encode categorical features
    if categorical_features:
        categorical_encoder = OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=False,
            drop='first',  # Avoid multicollinearity
            dtype=np.float64
        )
        transformers.append(('categorical', categorical_encoder, categorical_features))
    
    # Build the column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop any columns not explicitly handled
        verbose_feature_names_out=False
    )
    
    # Create the complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    return pipeline


def get_feature_names(pipeline, X_sample):
    """
    Extract feature names from a fitted pipeline.
    
    Args:
        pipeline: Fitted sklearn pipeline
        X_sample: Sample input data
        
    Returns:
        list: Feature names after transformation
    """
    try:
        # Try to get feature names from the pipeline
        if hasattr(pipeline, 'get_feature_names_out'):
            return pipeline.get_feature_names_out()
        
        # Fallback: manually construct feature names
        feature_names = []
        
        # Get the preprocessor from pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'geo_cluster':
                for i in range(transformer.n_clusters):
                    feature_names.append(f'geo_cluster_{i}')
            elif name == 'depth_bin':
                feature_names.append('depth_bin')
            elif name == 'scaler':
                feature_names.extend([f'scaled_{col}' for col in columns])
            elif name == 'categorical':
                if hasattr(transformer, 'get_feature_names_out'):
                    cat_names = transformer.get_feature_names_out(columns)
                    feature_names.extend(cat_names)
                else:
                    feature_names.extend([f'cat_{col}' for col in columns])
        
        return feature_names
        
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        # Return generic names based on transformed shape
        n_features = pipeline.transform(X_sample[:1]).shape[1]
        return [f'feature_{i}' for i in range(n_features)]


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional interaction features for earthquake prediction.
    
    Args:
        df (pd.DataFrame): Input dataframe with geographic features
        
    Returns:
        pd.DataFrame: DataFrame with additional interaction features
    """
    df_enhanced = df.copy()
    
    # Distance from equator (latitude effect)
    df_enhanced['abs_latitude'] = np.abs(df_enhanced['latitude'])
    
    # Depth categories based on seismological knowledge
    df_enhanced['depth_category'] = pd.cut(
        df_enhanced['depth'], 
        bins=[0, 70, 300, 700], 
        labels=['shallow', 'intermediate', 'deep']
    )
    
    # Regional activity indicators (if we have temporal data)
    if 'year' in df_enhanced.columns:
        df_enhanced['decade'] = (df_enhanced['year'] // 10) * 10
    
    # Coordinate interactions
    df_enhanced['lat_lon_interaction'] = df_enhanced['latitude'] * df_enhanced['longitude']
    
    return df_enhanced

