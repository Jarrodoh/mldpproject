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




# feature_engineering.py (replace the whole file with this)

from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_feature_pipeline(available_columns=None):
    # Define which features are numeric/categorical
    numeric_features = [
        'latitude', 'longitude', 'depth', 'gap', 'rms', 'magNst',
        'year', 'month', 'day', 'day_of_week', 'hour', 'is_weekend'
    ]
    categorical_features = ['magType']  # Add more if you have more categorical features

    # Remove missing columns from lists (if not all are present)
    if available_columns is not None:
        numeric_features = [f for f in numeric_features if f in available_columns]
        categorical_features = [f for f in categorical_features if f in available_columns]

    # Binning for depth (can drop if you don’t want)
    transformers = [
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
    
    # No custom transformers!
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    return Pipeline([('preprocessor', preprocessor)])



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

