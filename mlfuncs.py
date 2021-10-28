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
from sklearn.model_selection import GridSearchCV
import sklearn.externals
import joblib


# import modules for machine learning models

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
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
    sgd_reg = SGDRegressor()
    # lin_reg = LinearRegression()
    ri = Ridge(random_state=SEED)
    lo = Lasso(max_iter = 5000, random_state=SEED)
    en = ElasticNet(random_state=SEED)
    dt_reg = DecisionTreeRegressor(random_state=SEED)
    rf_reg = RandomForestRegressor(random_state=SEED)

    # create tuple of regressors
    regressors = [
        ('SGDRegressor', sgd_reg),
        # ('LinearRegression', lin_reg),
        ('Ridge', ri),
        ('Lasso', lo),
        ('ElasticNet', en),
        ('DecisionTreeRegressor', dt_reg),
        ('RandomForestRegressor', rf_reg)
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
    best_regressor = max_score['Regressor'].values.tolist()[0]
    print("####################################################################################")
    print("{} model yielded the highest R Squared of: {:.2f}".format(max_score['Regressor'].values.tolist()[0], max_score['R Squared'].values.tolist()[0]))
    # plot scores
    plots.bar_plot(df_scores, 'Regressor', 'R Squared')
    return best_regressor

def regressor_hyperparameters(best_regressor):
    # regressor list
    regressor_list = ['SGDRegressor', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor', 'RandomForestRegressor']
    # param_dict
    params_dict = {
        'SGDRegressor': [
            {'model': SGDRegressor(random_state=SEED)},
            {'grid': {
            'sgdregressor__penalty': ['l2'],
            'sgdregressor__alpha': [0.0001, 0.001, 0.01, 1, 5, 10, 100, 1000],
            'sgdregressor__max_iter': [1000, 5000, 10000]
        }}],
        'Ridge': [
            {'model': Ridge(random_state=SEED)},
            {'grid': {'ridge__alpha': [0.0001, 0.001, 0.01, 1, 5, 10, 100, 1000]
        }}],
        'Lasso': [
            {'model': Lasso(random_state=SEED)},
            {'grid': {'lasso_aplha': [0.0001, 0.001, 0.01, 1, 5, 10, 100, 1000],
            'lasso__max_iter': [1000, 5000, 10000]
        }}],
        'ElasticNet': [
            {'model': ElasticNet(random_state=SEED)},
            {'grid': {'elasticnet__alpha': [0.0001, 0.001, 0.01, 1, 5, 10, 100, 1000]
        }}],
        'DecisionTreeRegressor': [
            {'model': DecisionTreeRegressor(random_state=SEED)},
            { 'grid': {'decisiontreeregressor__max_depth': [2, 4, 8, 10, 12, 16, 20],
            'decisiontreeregressor__min_samples_leaf': [2, 4, 8, 10, 12, 16, 20]
        }}],
        'RandomForestRegressor': [
            {'model': RandomForestRegressor(random_state=SEED)},
            {'grid' : {'randomforestregressor__max_depth': [2, 15, 22],
            'randomforestregressor__min_samples_leaf': [1, 2, 4],
            'randomforestregressor__min_samples_split': [2, 4, 6]
            # 'randomforestregressor__min_weight_fraction_leaf': [0.1, 0.3, 0.9],
            # 'randomforestregressor__max_features': ['auto', 'log2', 'sqrt', None]
            # 'randomforestregressor__max_leaf_nodes': [None, 10, 40, 90]
        }}]
    }

    if best_regressor in regressor_list:
        grid_params = params_dict[best_regressor][1]['grid']
        grid_model = params_dict[best_regressor][0]['model']
    else:
        print("{} is not among list of regressors.".format(best_regressor))
    
    return grid_params, grid_model

def best_regressor_hyperparameter(df, best_regressor):
    print("Hyperparameter Tuning for the Best Regressor: {}".format(best_regressor))
    # extract train and test data
    X_train, X_test, y_train, y_test = pre_process_split_data(df)
    # extract cat and num features
    cat_feats, num_feats = extract_cat_num(X_train)
    # instantiate preprocessor
    preprocessor = preprocess_col_transformer(cat_feats, num_feats)
    # extract best regressor model and grid params
    grid_params, grid_model = regressor_hyperparameters(best_regressor)
    # instantiate pipeline
    print("Creating pipeline for {}.".format(best_regressor))
    pipe = make_pipeline(preprocessor, grid_model)
    # dictionary to hold gridsearch model output
    GridSearchCV_model_output = {}
    # create gridsearch object
    grid_search = GridSearchCV(pipe, param_grid=grid_params, cv = 10, scoring = 'r2', refit = True, n_jobs = 4, return_train_score = True)
    # fit gridsearch to training data
    grid_search.fit(X_train, y_train)
    # create key, value pair for best regressor
    GridSearchCV_model_output['best_regressor_grid_object'] = grid_search.best_estimator_
    # get best params
    grid_best_params = grid_search.best_params_
    print("Best parameters after GridSearchCV:\n {}".format(grid_best_params))
    GridSearchCV_model_output['best_params'] = grid_best_params
    # best score
    grid_best_score = grid_search.best_score_
    print("Best score after GridSearchCV:\n {}".format(grid_best_score))
    # get feature names
    cat_feature_names = grid_search.best_estimator_.named_steps['columntransformer'].named_transformers_['pipeline-1'].\
        named_steps['onehotencoder'].get_feature_names(input_features = cat_feats)
    all_feature_names = np.r_[cat_feature_names, num_feats]
    # list to hold coefs or feature imporatnces in the case of decision trees and random forest
    coefs_or_feats_imp = []

    if not 'RandomForestRegressor' or not 'DecisionTreeRegressor':
        # grab the coefficients
        best_regressor_coef = list(grid_search.best_estimator_.named_steps[best_regressor.lower()].coef_)
        best_regressor_coef_x = [x for x in best_regressor_coef]
        coefs_or_feats_imp.append(best_regressor_coef_x)
        # grab the intercept
        grid_search_intercept = grid_search.best_estimator_.named_steps[best_regressor.lower()].intercept_
        GridSearchCV_model_output['best_regressor_intercept'] = grid_search_intercept
    else:
        # grab the coefficients
        best_regressor_coef = list(grid_search.best_estimator_.named_steps[best_regressor.lower()].feature_importances_)
        best_regressor_coef_x = [x for x in best_regressor_coef]
        coefs_or_feats_imp.append(best_regressor_coef_x)

    # create a dataframe of feature names and coeeficients
    coef_dict = dict(zip(all_feature_names, coefs_or_feats_imp[0]))
    df_coef = pd.DataFrame(coef_dict.items(), columns = ['Feature', 'Coefficient'])
    df_coef = df_coef.sort_values(by = ['Coefficient'], ascending = False)
    df_coef = df_coef.reset_index(drop = True)
    GridSearchCV_model_output['best_regressor_feature_importances'] = df_coef

    return GridSearchCV_model_output, plots.bar_plot(df_coef, 'Feature', 'Coefficient')

def dump_estimator(GridSearchCV_model_output, best_regressor):
    # regressor list
    regressor_list = ['SGDRegressor', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor', 'RandomForestRegressor']
    if best_regressor in regressor_list:
        best_estimator = GridSearchCV_model_output['best_regressor_grid_object']
        model = joblib.dump(best_estimator, 'models/best_regressor_dump.pkl')
    else:
        print("{} is not among list of regressors.".format(best_regressor))
    return model




        