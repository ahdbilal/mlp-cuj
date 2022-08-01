"""
This module defines the following routines used by the 'train' step of the regression pipeline:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model pipeline.
"""


def estimator_fn():
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    import xgboost
    
    return xgboost.XGBRegressor(
      colsample_bytree=0.4123960280921751,
      learning_rate=0.018535644061244888,
      max_depth=4,
      min_child_weight=1,
      n_jobs=100,
      subsample=0.37028280278649944,
      verbosity=0,
      random_state=206170845,
    )