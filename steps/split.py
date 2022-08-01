"""
This module defines the following routines used by the 'split' step of the regression pipeline:

- ``process_splits``: Defines customizable logic for processing & cleaning the training, validation,
  and test datasets produced by the data splitting procedure.
"""

from pandas import DataFrame
import numpy as np
from scipy.stats import skew
import pandas as pd


def process_splits(
    train_df: DataFrame, validation_df: DataFrame, test_df: DataFrame
) -> (DataFrame, DataFrame, DataFrame):
    """
    Perform additional processing on the split datasets.

    :param train_df: The training dataset produced by the data splitting procedure.
    :param validation_df: The validation dataset produced by the data splitting procedure.
    :param test_df: The test dataset produced by the data splitting procedure.
    :return: A tuple containing, in order: the processed training dataset, the processed
             validation dataset, and the processed test dataset.
    """

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

    return process(train_df), process(validation_df), process(test_df)