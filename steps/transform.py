"""
This module defines the following routines used by the 'transform' step of the regression pipeline:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin

def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    ordinal_columns = ['ExterQual','ExterCond','BsmtCond','BsmtQual','HeatingQC','KitchenQual','KitchenQual','FireplaceQu','GarageQual','GarageCond','CentralAir','LotShape',\
                 'BsmtExposure','BsmtFinType1','BsmtFinType1']
    
    one_hot_columns = ['GarageFinish', 'Condition1','PavedDrive','BsmtFinType2','SaleType','HouseStyle','Electrical','BldgType','MasVnrArea','Foundation','SaleCondition','LotConfig','Neighborhood','RoofStyle','LandContour','MSZoning','Exterior1st','GarageType','MasVnrType','LandSlope','Exterior2nd','Functional']
    
    numerical_columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold','Age_House', 'TotalBsmtBath',
       'TotalBath', 'TotalSA']
    
    
    return Pipeline(
        steps=[
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "ordinal_encoder",
                            OrdinalEncoder(handle_unknown = 'use_encoded_value',unknown_value=-1),
                            ordinal_columns,
                        ),
                        (
                            "one_hot_encoder",
                            OneHotEncoder(categories="auto", sparse=False, handle_unknown = 'ignore'),
                            one_hot_columns,
                        ),
                        (
                            "standard_scalar",
                            FunctionTransformer(lambda x:x),
                            numerical_columns,
                        ),
                    ]
                ),
            ),
        ]
    )