import feature_engineering_functions as fe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
from math import sqrt
import numpy



import numpy as np
import pandas as pd


def create(model: object, df: pd.DataFrame, nm_fts: list, drp_fts: list, cat_fts: list, target: str, num_scaler: MinMaxScaler, ohe: OneHotEncoder, poly: PolynomialFeatures, test_size=0.2, rnd_state=40) -> dict:
    """Creates a linear model

    Args:
        df (pd.DataFrame): The dataframe to create the model from
        nm_fts (list): The numeric features
        drp_fts (list): The drop features
        target (list): The target label
        test_size (float, optional): The test size to use. Defaults to 0.2.
        rnd_state (int, optional): The random test split to use. Defaults to 40.

    Returns:
        _type_: A dictionary containing the scores and the models
    """

    df = df.drop(drp_fts, axis=1)

    x = df.drop(target, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=rnd_state)


    # x_train Alone
    
    x_train = fe.engineer_df(x_train, nm_fts, cat_fts, num_scaler, poly, ohe, True)
    
    # x_test Alone
    
    x_test = fe.engineer_df(x_test, nm_fts, cat_fts, num_scaler, poly, ohe, False)
    
    # x Alone
    
    x = fe.engineer_df(x, nm_fts, cat_fts, num_scaler, poly, ohe, False)
    
    model.fit(x_train, np.ravel(y_train))
    
    test_scores = cross_val_score(model, x, np.ravel(y),cv=KFold(5,shuffle=True), scoring='r2')

    return {
        'x_val_av_min_std': test_scores.mean() - test_scores.std(),
        'x_val_av': test_scores.mean(),
        'x_val_std': test_scores.std(),
        'test_score': model.score(x_test, np.ravel(y_test)),
        'test_mse': mean_squared_error(y_test, model.predict(x_test)),
        'test_rmse': mean_squared_error(y_test, model.predict(x_test), squared=False),
        'test_rmsle': mean_squared_error(numpy.log10(y_test + 1), numpy.log10(model.predict(x_test) + 1), squared=False),
        'train_score': model.score(x_train, np.ravel(y_train)),
        'train_mse': mean_squared_error(y_train, model.predict(x_train)),
        'train_rmse': mean_squared_error(y_train, model.predict(x_train), squared=False),
        'model': model,
    }
