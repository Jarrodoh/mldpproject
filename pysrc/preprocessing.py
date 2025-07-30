"""
Module: preprocessing
Description: Clean and prepare earthquake data for machine learning.
Enhanced with parameterized bounds, metadata return, and duplicate handling.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def preprocess(
    df: pd.DataFrame,
    missing_thresh: float = 0.3,
    impute_strategy: str = 'median',
    drop_cols: list = None,
    mag_bounds: tuple = (0, 10),
    depth_bounds: tuple = (0, 700),
    return_metadata: bool = False
) -> pd.DataFrame:
    """
    Clean and prepare earthquake data for modeling.

    Steps:
    1. Remove duplicate rows
    2. Drop columns with >missing_thresh fraction of nulls
    3. Extract temporal features from datetime before dropping
    4. Impute remaining numeric nulls using specified strategy
    5. Remove unrealistic outliers based on domain knowledge
    6. Drop administrative/text columns

    Args:
        df (pd.DataFrame): Raw DataFrame.
        missing_thresh (float): Null fraction threshold to drop columns.
        impute_strategy (str): 'mean' or 'median'.
        drop_cols (list): Additional columns to drop.
        mag_bounds (tuple): (min, max) bounds for magnitude clipping.
        depth_bounds (tuple): (min, max) bounds for depth clipping.
        return_metadata (bool): If True, return preprocessing metadata.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
        dict (optional): Metadata about preprocessing steps if return_metadata=True.
    """
    initial_shape = df.shape
    metadata = {'initial_shape': initial_shape}
    
    # 1. Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")
    metadata['duplicates_removed'] = duplicates_removed

    # 2. Drop columns with too many missing values
    null_frac = df.isnull().mean()
    drop_many = null_frac[null_frac > missing_thresh].index.tolist()
    df = df.drop(columns=drop_many)
    metadata['dropped_columns'] = drop_many
    
    if drop_many:
        logger.info(f"Dropped {len(drop_many)} columns with >{missing_thresh:.0%} missing: {drop_many}")

    # 3. Date parsing and feature extraction (BEFORE dropping time column)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        df['day_of_week'] = df['time'].dt.dayofweek  # Monday=0, Sunday=6
        df['hour'] = df['time'].dt.hour
        
        # Create time-based features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        logger.info("Extracted temporal features: year, month, day, day_of_week, hour, is_weekend")

    # 4. Impute numeric columns (exclude target variable)
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(['mag'])
    imputed_cols = []
    
    for col in num_cols:
        if df[col].isnull().any():
            if impute_strategy == 'median':
                fill_value = df[col].median()
            else:
                fill_value = df[col].mean()
            df[col].fillna(fill_value, inplace=True)
            imputed_cols.append(col)
    
    metadata['imputed_columns'] = imputed_cols
    if imputed_cols:
        logger.info(f"Imputed {len(imputed_cols)} columns using {impute_strategy}: {imputed_cols}")

    # 5. Outlier handling: clip magnitude and depth to realistic bounds
    if 'mag' in df.columns:
        mag_clipped = ((df['mag'] < mag_bounds[0]) | (df['mag'] > mag_bounds[1])).sum()
        df['mag'] = df['mag'].clip(lower=mag_bounds[0], upper=mag_bounds[1])
        metadata['mag_outliers_clipped'] = mag_clipped
        
    if 'depth' in df.columns:
        depth_clipped = ((df['depth'] < depth_bounds[0]) | (df['depth'] > depth_bounds[1])).sum()
        df['depth'] = df['depth'].clip(lower=depth_bounds[0], upper=depth_bounds[1])
        metadata['depth_outliers_clipped'] = depth_clipped

    # 6. Drop administrative/text columns and time (after feature extraction)
    default_drop = ['id', 'place', 'url', 'detail', 'updated', 'status', 'net', 
                   'time', 'type', 'locationSource', 'magSource']  # Added time here
    cols_to_drop = drop_cols or []
    all_drop_cols = list(set(default_drop + cols_to_drop) & set(df.columns))
    df = df.drop(columns=all_drop_cols)
    metadata['admin_columns_dropped'] = all_drop_cols
    
    if all_drop_cols:
        logger.info(f"Dropped {len(all_drop_cols)} administrative columns")

    # Final statistics
    final_shape = df.shape
    metadata['final_shape'] = final_shape
    metadata['rows_removed'] = initial_shape[0] - final_shape[0]
    metadata['columns_removed'] = initial_shape[1] - final_shape[1]
    
    logger.info(f"Preprocessing complete: {initial_shape} → {final_shape}")
    logger.info(f"Removed {metadata['rows_removed']} rows, {metadata['columns_removed']} columns")

    # Ensure all numeric columns are float64 for downstream compatibility
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float64)
    if return_metadata:
        return df, metadata
    return df


def split_data(
    df: pd.DataFrame,
    target: str = 'mag',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_bins: int = None
):
    # Ensure all numeric columns are float64 for downstream compatibility
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float64)
    """
    Split cleaned DataFrame into train/test sets with optional stratification.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
        target (str): Target column name
        test_size (float): Fraction for test set
        random_state (int): Random seed for reproducibility
        stratify_bins (int): If provided, stratify by binned target values
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    # Optional stratification for regression targets
    stratify = None
    if stratify_bins:
        # Create bins for stratification
        y_binned = pd.cut(y, bins=stratify_bins, labels=False)
        stratify = y_binned
        logger.info(f"Using stratified split with {stratify_bins} bins")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    logger.info(f"Split data: Train={X_train.shape}, Test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def get_preprocessing_summary(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Generate a summary of preprocessing changes.
    
    Args:
        df_before (pd.DataFrame): DataFrame before preprocessing
        df_after (pd.DataFrame): DataFrame after preprocessing
        
    Returns:
        dict: Summary statistics
    """
    return {
        'shape_before': df_before.shape,
        'shape_after': df_after.shape,
        'rows_removed': df_before.shape[0] - df_after.shape[0],
        'columns_removed': df_before.shape[1] - df_after.shape[1],
        'memory_before_mb': df_before.memory_usage(deep=True).sum() / 1024**2,
        'memory_after_mb': df_after.memory_usage(deep=True).sum() / 1024**2,
        'columns_before': list(df_before.columns),
        'columns_after': list(df_after.columns),
        'missing_values_before': df_before.isnull().sum().sum(),
        'missing_values_after': df_after.isnull().sum().sum()
    }