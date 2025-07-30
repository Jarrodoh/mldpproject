# Project Title: Earthquake Magnitude Prediction

## Project Structure & File Descriptions

- `app.py`: Main Streamlit app for earthquake magnitude prediction, interactive data insights, model results, and project overview.
- `requirements.txt`: Lists all Python dependencies required to run the app and notebooks.
- `README.md`: Project documentation and file descriptions.
- `test_imports.py`: Simple test script to check if all required modules can be imported.

### Folders
- `pysrc/`: Modular Python code for the ML pipeline.
  - `__init__.py`: Makes `pysrc` a package.
  - `data_loader.py`: Functions for loading and parsing earthquake datasets.
  - `preprocessing.py`: Data cleaning, missing value handling, and outlier clipping.
  - `feature_engineering.py`: Feature creation and transformation logic.
  - `models.py`: Model training, evaluation, and selection routines.
  - `utils/`: Utility functions (may include helpers for logging, metrics, etc.).
  - `__pycache__/`: Python cache files (auto-generated).
- `data/`: Raw and processed data files.
  - `usgs_earthquake_data_2000_2025.csv`: Main dataset used for training and analysis from kaggle: https://www.kaggle.com/datasets/pulastya/global-seismic-events-20002025 . Actually the dataset here is around 10-15k rows short of the actual dataset i got form kaggle because of file size limitations imposed by github, so uploading is impossible for mroe than 25mb, mine was 31mb so I compromised by reducign the data here, but everything else is triained on 170k rows (the real dataset from kaggle).
- `notebooks/`: Jupyter notebooks for EDA and model development.
  - `01_data_exploration.ipynb`: Initial data exploration and visualization.
  - `best_model.joblib`: Saved trained model for use in the app.




---


## Notebook Workflow: Step-by-Step Project Process

The Jupyter notebook (`notebooks/01_data_exploration.ipynb`) documents the full machine learning pipeline for earthquake magnitude prediction. Here’s a summary of the steps:

### Step 1: Import Modules
- Imported all custom pipeline modules (`data_loader`, `preprocessing`, `feature_engineering`, `models`) and essential libraries for data science and visualization.

### Step 2: Load Data
- Loaded the USGS earthquake dataset from `data/usgs_earthquake_data_2000_2025.csv`.
- Inspected the shape, top rows, and key characteristics of the raw data.

### Step 3: Data Exploration
- Explored dataset statistics, missing values, magnitude types, temporal and geographic coverage.
- Summarized key observations and data quality issues.

### Step 4: Enhanced Preprocessing
- Cleaned the data: handled missing values, clipped outliers, removed duplicates, and dropped administrative columns.
- Added temporal features (year, month, day, hour, etc.) and ensured all numeric columns were float64.
- Summarized preprocessing changes and final feature set.

### Step 5: Model Training
- Split the cleaned data into training and test sets.
- Trained multiple models (Ridge, SVR, HistGradientBoosting, Random Forest) using modular pipelines that include feature engineering (geographic clustering, depth binning, scaling, encoding).
- Summarized cross-validation results and selected the best model based on CV score.

### Step 6: Model Evaluation
- Evaluated all models on the test set using regression metrics: RMSE, MAE, R², and accuracy within ±0.5 magnitude.
- Identified and reported the best test-set model.

### Step 7: Feature Importance Analysis
- Used tree-based models to extract and plot the top-10 most important features.
- Interpreted which features (depth, geographic cluster, temporal) drive predictions.

### Step 8: Residual Analysis & Diagnostics
- Visualized residuals, error distributions, Q–Q plots, and error by magnitude range.
- Compared model performance and diagnosed strengths/weaknesses.

### Step 9: Model Saving
- Saved the best trained model to `notebooks/best_model.joblib` for use in the Streamlit app and future predictions.

### Step 10: Interpretation & Next Steps
- Added markdown explanations for all results, including scientific interpretation, limitations, and suggestions for future work.

---

## How to Use
- Run `app.py` with Streamlit to launch the interactive app.
- Explore the full data science workflow and model results in `notebooks/01_data_exploration.ipynb`.
- See `requirements.txt` for dependencies.

---

## Contact
For questions or collaboration, contact the Temasek Y2 MLDP Team
