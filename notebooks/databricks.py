# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Regression Pipeline Databricks Notebook
# MAGIC This notebook runs the MLflow Regression Pipeline on Databricks and inspects its results.
# MAGIC 
# MAGIC For more information about the MLflow Regression Pipeline, including usage examples,
# MAGIC see the [Regression Pipeline overview documentation](https://mlflow.org/docs/latest/pipelines.html#regression-pipeline)
# MAGIC and the [Regression Pipeline API documentation](https://mlflow.org/docs/latest/python_api/mlflow.pipelines.html#module-mlflow.pipelines.regression.v1.pipeline).

# COMMAND ----------

# MAGIC %pip install mlflow[pipelines]
# MAGIC %pip install -r ../requirements.txt
# MAGIC %pip install hyperopt

# COMMAND ----------

from mlflow.pipelines import Pipeline

p = Pipeline(profile="databricks")

# COMMAND ----------

p.clean()

# COMMAND ----------

p.inspect()

# COMMAND ----------

p.run("ingest")

# COMMAND ----------

from pandas import DataFrame
import numpy as np
from scipy.stats import skew
import pandas as pd
def process(df: DataFrame):

    #log transform the target:
    df = df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)

    df['GarageYrBlt'] = df['GarageYrBlt'].apply(pd.to_numeric, args=('coerce',))
    df['LotFrontage'] = df['LotFrontage'].apply(pd.to_numeric, args=('coerce',))

    object_columns_df = df.select_dtypes(include=['object'])
    numerical_columns_df =df.select_dtypes(exclude=['object'])

    columns_None = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']
    object_columns_df[columns_None]= object_columns_df[columns_None].fillna('None')

    columns_with_lowNA = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']
    #fill missing values for each column (using its own most frequent value)
    object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(object_columns_df.mode().iloc[0])

    numerical_columns_df['GarageYrBlt'] = numerical_columns_df['GarageYrBlt'].fillna(numerical_columns_df['YrSold']-35)
    numerical_columns_df['LotFrontage'] = numerical_columns_df['LotFrontage'].fillna(68)
    numerical_columns_df= numerical_columns_df.fillna(0)

    object_columns_df = object_columns_df.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)


    numerical_columns_df['Age_House']= (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt'])
    numerical_columns_df['Age_House'].describe()

    Negatif = numerical_columns_df[numerical_columns_df['Age_House'] < 0]

    numerical_columns_df.loc[numerical_columns_df['YrSold'] < numerical_columns_df['YearBuilt'],'YrSold' ] = 2009
    numerical_columns_df['Age_House']= (numerical_columns_df['YrSold']-numerical_columns_df['YearBuilt'])

    numerical_columns_df['TotalBsmtBath'] = numerical_columns_df['BsmtFullBath'] + numerical_columns_df['BsmtFullBath']*0.5
    numerical_columns_df['TotalBath'] = numerical_columns_df['FullBath'] + numerical_columns_df['HalfBath']*0.5 
    numerical_columns_df['TotalSA']=numerical_columns_df['TotalBsmtSF'] + numerical_columns_df['1stFlrSF'] + numerical_columns_df['2ndFlrSF'] 

    df_final = pd.concat([object_columns_df, numerical_columns_df], axis=1,sort=False)

    return df_final

# COMMAND ----------

process(p.get_artifact("ingested_data"))

# COMMAND ----------

p.run("split")

# COMMAND ----------

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

# COMMAND ----------

transformer_fn().fit_transform(p.get_artifact("training_data"))

# COMMAND ----------

p.run("transform")

# COMMAND ----------

def estimator_fn(params):
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    import xgboost
    
    return xgboost.XGBRegressor(**params)

# COMMAND ----------

p.get_artifact("transformed_training_data").iloc[:,0:-1]

# COMMAND ----------

model = estimator_fn().fit(p.get_artifact("transformed_training_data").iloc[:,0:-1],p.get_artifact("transformed_training_data")["SalePrice"])

# COMMAND ----------

from sklearn.metrics import mean_squared_error
mean_squared_error(model.predict(p.get_artifact("transformed_validation_data").iloc[:,0:-1]), p.get_artifact("transformed_validation_data")["SalePrice"], squared=False)

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
import numpy as np
import mlflow

X_train = p.get_artifact("transformed_training_data").iloc[:,0:-1]
y_train = p.get_artifact("transformed_training_data")["SalePrice"]
X_val = p.get_artifact("transformed_validation_data").iloc[:,0:-1]
y_val = p.get_artifact("transformed_validation_data")["SalePrice"]

def train_model(params):
  mlflow.autolog()
  with mlflow.start_run(nested=True):
    # Create a support vector classifier model
    clf = xgboost.XGBRegressor(**params)
    clf.fit(X_train,y_train)
    
    # Use the cross-validation accuracy to compare the models' performance
    accuracy = mean_squared_error(y_val, clf.predict(X_val), squared=False)
    mlflow.log_metric('rmse', accuracy)
    
    # Hyperopt tries to minimize the objective function. A higher accuracy value means a better model, so you must return the negative accuracy.
    return {'loss': -accuracy,'status': STATUS_OK}

random_state = 314159265
search_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    # A problem with max_depth casted to float instead of int with
    # the hp.quniform method.
    'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    # Increase this number if you have more cores. Otherwise, remove it and it will default 
    # to the maxium number. 
    'n_jobs': 100,
    'silent': 1,
    'seed': random_state
}

spark_trials = SparkTrials()
 
with mlflow.start_run(run_name='xb_hyperopt') as run:
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=32,
    trials=spark_trials)

# COMMAND ----------

  p.run("train")

# COMMAND ----------

p.run("evaluate")

# COMMAND ----------

p.run("register")

# COMMAND ----------

p.inspect("train")

# COMMAND ----------

test_data = p.get_artifact("test_data")
test_data.describe()

# COMMAND ----------

trained_model = p.get_artifact("model")
print(trained_model)
