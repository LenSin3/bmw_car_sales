# import libraries for data loading, cleaning and transformation
import os
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import utils

## function to read data
## check for data quality issues

def read_and_qa(pth_str):
    # check if path exists
    if os.path.exists(pth_str):
        # read file
        df = pd.read_csv(pth_str)
        # copy data frame
        df = df.copy()
        # check for duplicates
        print("Checking for duplicate rows in the dataframe...")
        row_dupes = df.duplicated()
        if sum(row_dupes) == 0:
            print("There are no duplicate rows in dataframe.")
        elif sum(row_dupes) > 0:
            print("Dropping duplicate rows in dataframe.")
            # drop duplicates
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)

        ##############################
        # check for nulls
        data_types = []
        non_nulls = []
        nulls = []
        null_column_percent = []
        null_df_percent = []
        df_cols = df.columns
        print("There are {} columns and {} records in the dataframe.".format(len(df_cols), len(df)))
        # loop through columns and capture the variables above
        print("Extracting count and percentages of nulls and non nulls")
        for col in df_cols:
            # extract null count
            null_count = df[col].isna().sum()
            nulls.append(null_count)
                
            # extract non null count
            non_null_count = len(df) - null_count
            non_nulls.append(non_null_count)
                
            # extract % of null in column
            col_null_perc = 100 * null_count/len(df)
            null_column_percent.append(col_null_perc)
            
            if null_count == 0:
                null_df_percent.append(0)
            else:
                # extract % of nulls out of total nulls in dataframe
                df_null_perc = 100 * null_count/df.isna().sum().sum()
                null_df_percent.append(df_null_perc)
                
            # capture data types
            data_types.append(df[col].dtypes) 
    
    else:
        raise utils.InvalidFilePath(pth_str)
            
    # create zipped list with column names, data_types, nulls and non nulls
    lst_data = list(zip(df_cols, data_types, non_nulls, nulls, null_column_percent, null_df_percent))
    # create dataframe of zipped list
    df_zipped = pd.DataFrame(lst_data, columns = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsInColumn', 'PercentOfNullsInData'])
    return df, df_zipped


def unique_vals_counts(df: DataFrame) -> DataFrame:
    """Count unique values in dataframe.

    The count is perfromed on all columns in the dataframe.

    Parameters
    ----------
    df : DataFrame
        Dataframe to check for unique values per column.
        

    Returns
    -------
    DataFrame
    """
    if isinstance(df, pd.DataFrame):
        vals = df.nunique().reset_index()
    else:
        raise utils.InvalidDataFrame(df)
    return vals.rename(columns = {'index': 'column', 0: 'count'})

def unique_vals_column(df: DataFrame, col: str, normalize = False) -> DataFrame:
    """Count unique values in a single column in a dataframe.

    Value counts are calculated for a single column and tabulated.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing column to check 
        for unique values.
    col : str
        Name of column to check for unique values.
    normalized : bool, optional
        If true this function normalizes the counts.
         (Default value = False)
         

    Returns
    -------
    DataFrame
    """
    if isinstance(df, pd.DataFrame):
        # check if column in dataframe
        if col in df.columns:
            # get unique value counts of column
            uniques = df[col].value_counts().reset_index().rename(columns = {'index': col, col : 'count'})
            if normalize:
                uniques = df[col].value_counts(normalize = True).reset_index().rename(columns = {'index': col, col : 'percentOfTotal'})
        else:
            raise utils.InvalidColumn(col)
    else:
        raise utils.InvalidDataFrame(df)
    return uniques

# function to clean data
def bmw_data_clean(df):
    # list of bmw columns
    bmw_cols = ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']
    if isinstance(df, pd.DataFrame):
        # get df columns
        df_cols = df.columns.tolist()
        # check for membership in bmw_cols
        bmw_membership = any(col in bmw_cols for col in df_cols)
        if bmw_membership:
            # strip models of leading spaces
            df['model'] = df['model'].apply(lambda label: label.strip())
            # get age of each car with respect to max date
            df['num_year'] = [df['year'].max() - x for x in df['year']]
        else:
            raise utils.UnexpectedDataFrame(df)
    else:
        raise utils.InvalidDataFrame(df)
    
    return df

