import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from typing import Callable, List, Any, Optional, Tuple
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator

def overunder_error(y_true, y_pred, underpred_penalty=1.0, overpred_penalty=1.0, alpha=0.5):
    """
    Calculate the Over-Under Error for portfolio optimization.

    Parameters:
        y_true (array-like): True portfolio values.
        y_pred (array-like): Predicted portfolio values.
        underpred_penalty (float, optional): Penalty factor for underpredictions. Default is 1.0.
        overpred_penalty (float, optional): Penalty factor for overpredictions. Default is 1.0.
        alpha (float, optional): Exponent for residual calculation. Default is 0.5.

    Returns:
        float: Mean Over-Under Error.

    The Over-Under Error is a custom loss function written for portfolio optimization. It calculates a loss based on the 
    differences between true and predicted portfolio values. The function allows for penalties on overpredictions and 
    underpredictions using the 'overpred_penalty' and 'underpred_penalty' parameters, respectively. The 'alpha' parameter
    determines the exponent for the residual calculation.

    If the residual (y_true - y_pred) is negative, it represents an underprediction, and the underprediction penalty is
    applied. If the residual is positive, it represents an overprediction, and the overprediction penalty is applied.

    The function returns the mean Over-Under Error, which can be used as an optimization objective in portfolio management.
    """
    # Calculate the residual between true and predicted values
    residual = y_true - y_pred
    
    # Define the loss using numpy's where function
    loss = np.where(
        residual < 0,  # If residual is negative (underprediction)
        underpred_penalty * np.abs(residual)**alpha,  # Apply underprediction penalty
        overpred_penalty * np.abs(residual)**alpha  # Apply overprediction penalty
    )
    
    # Calculate the mean of the loss
    return np.mean(loss)

def ts_cross_val_score(model: BaseEstimator, 
                       X: DataFrame, 
                       y: Series, 
                       cv: int, 
                       scorer: Callable, 
                       **scorer_kwargs: Optional[Any]) -> List[float]:
    """
    Perform time-series cross-validation on a given model using a specific scoring function.
    
    Parameters:
    - model (BaseEstimator): The machine learning model to be trained and validated.
    - X_train (DataFrame): Feature matrix for the training data.
    - y_train (Series): Target vector for the training data.
    - cv (int): Number of splits/folds for cross-validation.
    - scorer (Callable): Scoring function to evaluate the predictions. Must take two arrays 
                         'y_true' and 'y_pred' as arguments, along with any additional 
                         keyword arguments (**scorer_kwargs).
    - **scorer_kwargs (Optional[Any]): Additional keyword arguments to pass to the scoring function.
        
    Returns:
    - cv_scores (List[float]): List of scores calculated for each fold during cross-validation.
    """
    tscv = TimeSeriesSplit(cv)
    cv_scores = []
    
    for idx_train, idx_test in tscv.split(X):
        cvx_train, cvy_train = X.iloc[idx_train], y[idx_train]
        cvx_test, cvy_test = X.iloc[idx_test], y[idx_test]
        
        model.fit(cvx_train, cvy_train)
        cvy_hat = model.predict(cvx_test)
        
        score = scorer(cvy_test, cvy_hat, **scorer_kwargs)
        cv_scores.append(score)
    
    return cv_scores

def arimax_cross_val_score(X: DataFrame, 
                                 y: Series, 
                                 order: Tuple[int, int, int],
                                 pca: Pipeline,
                                 cv: int, 
                                 scorer: Callable,
                                 **scorer_kwargs: Optional[Any]) -> List[float]:
    """
    Perform time-series cross-validation using an ARIMA model with exogenous variables and PCA.

    Parameters:
        - X (DataFrame): Feature matrix.
        - y (Series): Target variable.
        - order (Tuple[int, int, int]): The (p,d,q) order of the model for the number of AR, differences, and MA parameters.
        - seasonal_order (Tuple[int, int, int, int]): The (P,D,Q,S) order of the seasonal component.
        - pca (Pipeline): PCA pipeline for dimensionality reduction.
        - cv (int): Number of folds in TimeSeriesSplit.
        - scorer (Callable): Custom scoring function.
        - **scorer_kwargs (Optional[Any]): Additional keyword arguments for the scoring function.

    Returns:
        List[float]: List of scores obtained for each fold during cross-validation.

    This function performs time-series cross-validation on an ARIMA model with exogenous variables,
    after reducing the dimensions of the exogenous variables using PCA. The function takes the time 
    series target variable 'y', feature matrix 'X', model orders, PCA model, number of CV folds, and 
    a scoring function. It returns a list of scores computed by the scoring function for each fold.
    """
    
    tscv = TimeSeriesSplit(n_splits=cv)
    cv_scores = []

    for train_idx, test_idx in tscv.split(y):
        cvx_train, cvy_train = X.iloc[train_idx], y[train_idx]
        cvx_test, cvy_test = X.iloc[test_idx], y[test_idx]

        cvx_train_pca = pca.fit_transform(cvx_train)
        cvx_test_pca = pca.transform(cvx_test)

        cv_model = SARIMAX(cvy_train.values, exog=cvx_train_pca, order=order).fit()
        cvy_hat = cv_model.predict(start=test_idx[0], end=test_idx[-1], exog=cvx_test_pca)

        score = scorer(cvy_test, cvy_hat, **scorer_kwargs)
        cv_scores.append(score)

    return cv_scores

def naive_cross_val_score(r_hat: Series, 
                          r_true: Series, 
                          cv: int, 
                          scorer: Callable, 
                          **scorer_kwargs: Optional[Any]) -> List[float]:
    """
    Perform time-series cross-validation using a naive forecast model.

    Parameters:
        - r_hat (Series): Predicted target values.
        - r_true (Series): Actual target values.
        - cv (int): Number of splits/folds for time-series cross-validation.
        - scorer (Callable): Scoring function to evaluate the predictions. 
                             Must take two arrays 'y_true' and 'y_pred' as arguments,
                             along with any additional keyword arguments (**scorer_kwargs).
        - **scorer_kwargs (Optional[Any]): Additional keyword arguments to pass to the scoring function.

    Returns:
        - cv_scores (List[float]): List of scores calculated for each fold during cross-validation.

    This function performs time-series cross-validation on a naive forecast model. It employs a user-defined
    scoring function to evaluate the model's predictions. The scoring function can be customized through additional
    keyword arguments, allowing for flexible evaluation metrics.
    """
    
    tscv = TimeSeriesSplit(n_splits=cv)
    cv_scores = []
    
    for train_idx, test_idx in tscv.split(r_true):
        _, act = r_true.iloc[train_idx], r_true.iloc[test_idx]
        _, pred = r_hat.iloc[train_idx], r_hat.iloc[test_idx]
        
        cv_scores.append(scorer(act, pred, **scorer_kwargs))
    
    return cv_scores