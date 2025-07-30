"""
Test script to verify all modules work correctly
"""
import sys
import os

# Add the project root to Python path so we can import from .pysrc
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

"""
Test script to verify all modules work correctly
"""
import sys
import os

# Add the .pysrc folder to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
pysrc_path = os.path.join(project_root, '.pysrc')
sys.path.insert(0, pysrc_path)

# Now we can import our modules directly
try:
    import data_loader
    print("✓ Successfully imported data_loader")
    
    # Test the function
    load_data = data_loader.load_data
    print("✓ load_data function available")
except ImportError as e:
    print(f"✗ Error importing data_loader: {e}")

try:
    import models
    print("✓ Successfully imported models")
    
    # Test the functions
    train_models = models.train_models
    evaluate_model = models.evaluate_model
    print("✓ train_models and evaluate_model functions available")
except ImportError as e:
    print(f"✗ Error importing models: {e}")

# Test data loading
try:
    # Adjust this path to your actual CSV file
    data_path = "data/usgs_earthquake_data_2000_2025.csv"
    if os.path.exists(data_path):
        df = load_data(data_path)
        print(f"✓ Successfully loaded data: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print(f"✗ Data file not found: {data_path}")
except Exception as e:
    print(f"✗ Error loading data: {e}")

print("\nTo use in Python REPL:")
print("1. Exit the REPL (type 'exit()')")
print("2. Run: python test_imports.py")
print("3. Or use: python -c 'exec(open(\"test_imports.py\").read())'")
