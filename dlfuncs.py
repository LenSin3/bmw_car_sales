import os
import numpy as np
import pandas as pd
import funcs
import mlfuncs
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


SEED = 42
# function to read and preprocess data for deep learning
def read_preprocess_data(df):
    if isinstance(df, pd.DataFrame):
        X = df.drop(['year', 'price'], axis=1)
        y = df['price']
    else:
        raise utils.InvalidDataFrame(df)
    return X, y

def split_data(X, y, test_size = 0.3, random_state = SEED):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    return X_train, X_test, y_train, y_test

def test_output(dfx, dfy):
    from sklearn.pipeline import make_pipeline
    X_train, X_test, y_train, y_test = split_data(dfx, dfy, test_size = 0.3, random_state = 42)
    cat_feats, num_feats = funcs.extract_cat_num(X_train)
    preprocessor = funcs.preprocess_col_transformer(cat_feats, num_feats)
    pipe = make_pipeline(preprocessor)
    X_td = pipe.fit_transform(X_train)
    X_tdt = pipe.transform(X_test)
    return X_td, X_tdt, y_train, y_test, pipe
        
def reg_model(n_cols, optimizer, loss):
    model = Sequential()
    model.add(Dense(100, input_shape = (n_cols,), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer=optimizer, loss=loss)
    return model

def train_model(df, optimizer, loss):
    dfx = df.drop(['year', 'price'], axis = 1)
    dfy = df['price']
    X_train, X_test, y_train, y_test, pipe = test_output(dfx, dfy)
    n_cols = X_train.shape[1]
    model = reg_model(n_cols, optimizer, loss)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 100)
    result = model.evaluate(X_test, y_test)
    y_preds = model.predict(X_test)
    return result, y_test, y_preds, model, pipe

