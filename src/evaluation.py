import numpy as np
import pandas as pd

import src.constants as cst
import src.pipeline as pipe
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def fit_and_evaluate_pipeline(estimator, features_dict: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Create a pipeline from an estimator and encoder, using a dictionary of feature preprocessing.
    Fit the pipeline and evaluate it on a test set.

    Args:
        estimator: model object
        features_dict (dict): dictionary of feature preprocessing
        X_train (pd.DataFrame): X_train
        y_train (pd.DataFrame): y_train
        X_test (pd.DataFrame): X_test
        y_test (pd.DataFrame): y_test

    Returns:
        dict: dictionary of evaluation metrics (RMSE and MAPE)
    """
    # Create a pipeline
    features = pipe.get_features_from_dict(features_dict)
    pipeline = pipe.build_pipeline(estimator, features_dict)

    # Fit the pipeline on the train set
    pipeline.fit(X_train[features], y_train)

    # Evaluate the pipeline on the test set and store results
    metrics = evaluate_model(pipeline, X_test[features], y_test)
    return metrics

def evaluate_model(estimator, X_test_encoded: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a model/pipeline on a test set and return metrics.

    Args:
        estimator: model object
        X_test_encoded (pd.DataFrame | np.array): X_test_encoded
        y_test (pd.Series): y_test

    Returns:
        dict: dictionary of evaluation metrics (RMSE and MAPE)
    """
    predictions = estimator.predict(X_test_encoded)
    metrics = {
        'mape': mean_absolute_percentage_error(y_test, predictions), 
        'rmse': np.sqrt(mean_squared_error(y_test, predictions))
    }
    return metrics

def split_train_test(data: pd.DataFrame):
    """Split the data into train/test and get the target_col (`valeur m2`) and initial target (`valeur').

    Args:
        data (pd.DataFrame): train data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=[cst.raw_target_col, cst.target_col]), 
        data[cst.target_col], 
        train_size=cst.train_size,
        random_state=cst.random_seed
        )

    y_train_valeur = data.loc[X_train.index, cst.raw_target_col]
    y_test_valeur = data.loc[X_test.index, cst.raw_target_col]

    assert X_train.shape[0]==y_train.shape[0]==y_train_valeur.shape[0]
    assert X_test.shape[0]==y_test.shape[0]==y_test_valeur.shape[0]
    
    return X_train, X_test, y_train, y_test, y_train_valeur, y_test_valeur