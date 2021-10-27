# import base libraries and dependencies
from re import UNICODE
from numpy.lib import utils
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import funcs
import plots
import utils

# import machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# import modules for machine learning models

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


SEED = 42

def pre_process_split_data(df):
    if isinstance(df, pd.DataFrame):
        # drop all outliers outside 3 standard deviations of price
        df = df[np.abs(df['price'] - df['price'].mean()) <= (3 * df['price'].std())]
        df.reset_index(drop = True, inplace = True)
        # drop price and year
        X = df.drop(['price', 'year'], axis=1)
        y = df['price']
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = SEED)
    else:
        raise utils.InvalidDataFrame(df)
    return X_train, X_test, y_train, y_test

def extract_cat_num(X_train):
    if isinstance(X_train, pd.DataFrame):
        # list to hold categorical and numerical column names
        cat_feats = []
        num_feats = []
        # extract column names
        xtrain_cols = X_train.columns.tolist()
        for col in xtrain_cols:
            if X_train[col].dtypes == 'object':
                cat_feats.append(col)
            elif X_train[col].dtypes == 'int64' or X_train[col].dtypes == 'float64' or X_train[col].dtypes == 'int32':
                num_feats.append(col)
    else:
        raise utils.InvalidDataFrame(X_train)
    return cat_feats, num_feats

def preprocess_col_transformer(cat_feats, num_feats):
    # create instances for imputation and encoding of categorical variables
    cat_imp = SimpleImputer(strategy = 'constant', fill_value = 'missing')
    ohe = OneHotEncoder(handle_unknown = 'ignore')
    cat_pipeline = make_pipeline(cat_imp, ohe)

    # create instances for imputation and encoding of numerical variables
    num_imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    std = StandardScaler()
    num_pipeline = make_pipeline(num_imp, std)

    # create a preprocessor object
    preprocessor = make_column_transformer(
        (cat_pipeline, cat_feats),
        (num_pipeline, num_feats),
        remainder = 'passthrough'
    )

    return preprocessor

def run_multi_models(df):
    # extract train and test data
    X_train, X_test, y_train, y_test = pre_process_split_data(df)
    # extract cat and num features
    cat_feats, num_feats = extract_cat_num(X_train)
    # instantiate preprocessor
    preprocessor = preprocess_col_transformer(cat_feats, num_feats)
    # instantiate regression models with base parameters
    lin_reg = LinearRegression()
    ri = Ridge(random_state=SEED)
    lo = Lasso(max_iter = 5000, random_state=SEED)
    en = ElasticNet(random_state=SEED)
    dt_reg = DecisionTreeRegressor(random_state=SEED)
    rf_reg = RandomForestRegressor(random_state=SEED)

    # create tuple of regressors
    regressors = [
        ('Linear Regression', lin_reg),
        ('Ridge', ri),
        ('Lasso', lo),
        ('ElasticNet', en),
        ('Decision Tree Regressor', dt_reg),
        ('Random Forest Regressor', rf_reg)
    ]

    # dictionary to hold accuracy scores
    acc_scores = {}

    for reg_name, reg in regressors:
        # instantiate pipeline
        print("Creating pipeline for {}.".format(reg_name))
        pipe = make_pipeline(preprocessor, reg)
        # fit training data to pipe
        print("Fitting training data to pipeline for {}.".format(reg_name))
        pipe.fit(X_train, y_train)
        # get predictions
        print("Predicting test values for {}.".format(reg_name))
        y_pred = pipe.predict(X_test)
        # get accuracy score
        print("Calculating accuracy score for {}.".format(reg_name))
        reg_acc_scr = r2_score(y_test, y_pred)
        print("Accuracy Score for {}: {}".format(reg_name, reg_acc_scr))
        # create key, value pair in acc_scores
        acc_scores[reg_name] = reg_acc_scr

    # create dataframe of acc_scores dict
    df_scores = pd.DataFrame(acc_scores.items(), columns=['Regressor', 'R Squared'])
    max_score = df_scores.loc[df_scores['R Squared'] == df_scores['R Squared'].max()]
    print("####################################################################################")
    print("{} model yielded the highest R Squared of: {:.2f}".format(max_score['Regressor'].values.tolist()[0], max_score['R Squared'].values.tolist()[0]))
    # plot scores
    return plots.bar_plot(df_scores, 'Regressor', 'R Squared')

def regressor_hyperparameters(regressor):
    

def best_regressor_hyperparameter(df, regressor):

        