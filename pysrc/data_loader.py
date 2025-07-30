"""
Module: data_loader
Description: Load and optimize earthquake data from CSV files.
Enhanced with dynamic path resolution, better logging, and sample loading.
"""
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(
    filepath: str = None,
    dtype_map: dict = None,
    parse_dates: list = ['time'],
    memory_optimize: bool = True
) -> pd.DataFrame:
    """
    Load earthquake data from CSV with automatic path resolution and optimization.
    
    Args:
        filepath (str): Path to CSV file. If None, uses default project data path.
        dtype_map (dict): Column-to-dtype mapping for memory/file performance.
        parse_dates (list): Columns to parse as datetime.
        memory_optimize (bool): If True, downcast numeric dtypes to reduce memory.
    
    Returns:
        pd.DataFrame: Loaded and optionally type-optimized DataFrame.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Dynamic path resolution for project portability
    if filepath is None:
        # Assume we're in pysrc/, go up to project root, then to data/
        BASE = Path(__file__).parent.parent
        filepath = BASE / "data" / "usgs_earthquake_data_2000_2025.csv"
    
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    logger.info(f"Loading data from: {path}")

    # Load CSV with dtype hints and date parsing
    df = pd.read_csv(filepath,
                     dtype=dtype_map,
                     parse_dates=parse_dates,
                     low_memory=False)

    # Log initial data info
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    logger.info(f"Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Memory optimization: downcast numeric types
    if memory_optimize:
        original_memory = df.memory_usage(deep=True).sum()
        df = optimize_dtypes(df)
        optimized_memory = df.memory_usage(deep=True).sum()
        memory_saved = (original_memory - optimized_memory) / 1024**2
        logger.info(f"Memory saved: {memory_saved:.2f} MB ({memory_saved/117.68*100:.1f}% reduction)")
    
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types and categorizing strings.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Memory-optimized DataFrame
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        if df_optimized[col].dtype == 'float64':
            # Try to downcast to float32
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        elif df_optimized[col].dtype == 'int64':
            # Try to downcast to smaller int
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        elif df_optimized[col].dtype == 'object':
            # Convert to category if beneficial (less than 50% unique values)
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            if num_unique / num_total < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
                logger.info(f"Converted {col} to category ({num_unique} unique values)")
    
    return df_optimized


def load_sample_data(filepath: str = None, sample_size: int = 10000, 
                    random_state: int = 42) -> pd.DataFrame:
    """
    Load a random sample of data for quick prototyping.
    
    Args:
        filepath (str): Path to CSV file
        sample_size (int): Number of rows to sample
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Sample of the full dataset
    """
    df = load_data(filepath, memory_optimize=False)  # Load full data first
    
    if len(df) <= sample_size:
        logger.warning(f"Dataset has only {len(df)} rows, returning full dataset")
        return optimize_dtypes(df)
    
    sample_df = df.sample(n=sample_size, random_state=random_state)
    logger.info(f"Sampled {sample_size:,} rows from {len(df):,} total rows")
    
    return optimize_dtypes(sample_df)
