"""
Module: models (full-data, expanded grid)
Description: Train and evaluate machine learning models for earthquake magnitude prediction
using the **entire** training set and richer hyperparameter grids.
"""
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from pysrc.feature_engineering import build_feature_pipeline
import logging

logger = logging.getLogger(__name__)

def train_models(X_train, y_train, fast_mode=False):
    """
    Train regression models on the FULL training set with expanded hyperparameter search.
    
    Args:
        X_train: pd.DataFrame of features (~140k rows)
        y_train: pd.Series of targets
        fast_mode (bool): if True, uses reduced grids & 3-fold CV; else full grids & 5-fold CV.
        
    Returns:
        dict of trained GridSearchCV objects.
    """
    models = {}
    available_columns = X_train.columns.tolist()
    cv_folds = 3 if fast_mode else 5
    
    print(f"🔧 Training with {cv_folds}-fold CV on {len(X_train):,} rows")
    print("📊 Model lineup:")
    print("  • RidgeRegression")
    print("  • LinearSVR")
    print("  • HistGradientBoostingRegressor")
    print("  • RandomForestRegressor\n")
    
    # 1) Ridge Regression
    ridge_pipe = Pipeline([
        ('pre', build_feature_pipeline(available_columns=available_columns)),
        ('ridge', Ridge(random_state=42))
    ])
    ridge_params = {'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}  # richer grid
    ridge_gs = GridSearchCV(
        ridge_pipe, ridge_params,
        cv=cv_folds, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    ridge_gs.fit(X_train, y_train)
    models['ridge'] = ridge_gs
    print(f"✅ Ridge done ({len(ridge_params['ridge__alpha'])}×{cv_folds} fits)")

    # 2) Linear SVR
    lsvr_pipe = Pipeline([
        ('pre', build_feature_pipeline(available_columns=available_columns)),
        ('lsvr', LinearSVR(random_state=42, max_iter=20000))
    ])
    lsvr_params = {'lsvr__C': [0.1, 1.0, 10.0, 100.0]}
    lsvr_gs = GridSearchCV(
        lsvr_pipe, lsvr_params,
        cv=cv_folds, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    lsvr_gs.fit(X_train, y_train)
    models['linear_svr'] = lsvr_gs
    print(f"✅ LinearSVR done ({len(lsvr_params['lsvr__C'])}×{cv_folds} fits)")

    # 3) HistGradientBoostingRegressor
    hgb_pipe = Pipeline([
        ('pre', build_feature_pipeline(available_columns=available_columns)),
        ('hgb', HistGradientBoostingRegressor(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ))
    ])
    hgb_params = {
        'hgb__max_iter': [100, 200, 300],
        'hgb__learning_rate': [0.01, 0.1],
        'hgb__max_depth': [None, 10, 20]
    }
    hgb_gs = GridSearchCV(
        hgb_pipe, hgb_params,
        cv=cv_folds, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    hgb_gs.fit(X_train, y_train)
    models['hist_gradient_boosting'] = hgb_gs
    total_hgb = (len(hgb_params['hgb__max_iter']) *
                 len(hgb_params['hgb__learning_rate']) *
                 len(hgb_params['hgb__max_depth']) *
                 cv_folds)
    print(f"✅ HistGradientBoosting done ({total_hgb} fits)")

    # 4) Random Forest
    rf_pipe = Pipeline([
        ('pre', build_feature_pipeline(available_columns=available_columns)),
        ('rf', RandomForestRegressor(
            random_state=42,
            n_jobs=1,
            oob_score=True
        ))
    ])
    rf_params = {
        'rf__n_estimators': [25, 50,],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2]
    }
    rf_gs = GridSearchCV(
        rf_pipe, rf_params,
        cv=cv_folds, scoring='neg_root_mean_squared_error',
        n_jobs=1, verbose=1
    )
    rf_gs.fit(X_train, y_train)
    models['random_forest'] = rf_gs
    total_rf = (len(rf_params['rf__n_estimators']) *
                len(rf_params['rf__max_depth']) *
                len(rf_params['rf__min_samples_split']) *
                len(rf_params['rf__min_samples_leaf']) *
                cv_folds)
    print(f"✅ RandomForest done ({total_rf} fits)")

    print(f"\n🎯 Total fits: {sum([m.cv_results_['params'].__len__() * cv_folds for m in models.values()])}")
    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'rmse': root_mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'within_0.5': (abs(y_pred - y_test) <= 0.5).mean()
    }


def compare_models(models_dict, X_test, y_test):
    results = {}
    for name, gs in models_dict.items():
        print(f"Evaluating {name}…")
        res = evaluate_model(gs.best_estimator_, X_test, y_test)
        res['cv_score'] = gs.best_score_
        res['best_params'] = gs.best_params_
        results[name] = res
    return results

def get_feature_importance(model, feature_names=None):
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained model (pipeline or estimator)
        feature_names: List of feature names
        
    Returns:
        dict: Feature importance scores (if available)
    """
    try:
        # Handle pipeline case
        if hasattr(model, 'named_steps'):
            estimator = model.named_steps[list(model.named_steps.keys())[-1]]
        else:
            estimator = model
            
        # Get feature importance for tree-based models
        if hasattr(estimator, 'feature_importances_'):
            importance = estimator.feature_importances_
            
            if feature_names is not None:
                return dict(zip(feature_names, importance))
            else:
                return {'feature_importance': importance.tolist()}
                
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    return None