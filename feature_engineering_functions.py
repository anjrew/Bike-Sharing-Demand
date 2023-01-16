import string
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


def find_season(month, hemisphere='Northern'):
    """Finds the season with the given month number

    Args:
        month (_type_): The month number
        hemisphere (str, optional): North or south of the equator. Defaults to 'Northern'.

    Returns:
        _type_: The season string
    """
    if hemisphere == 'Southern':
        season_month_south = {
            12: 'Summer', 1: 'Summer', 2: 'Summer',
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
            6: 'Winter', 7: 'Winter', 8: 'Winter',
            9: 'Spring', 10: 'Spring', 11: 'Spring'}
        return season_month_south.get(month)

    elif hemisphere == 'Northern':
        season_month_north = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        return season_month_north.get(month)
    else:
        print('Invalid selection. Please select a hemisphere and try again')


def find_time_of_day(hour: int) -> string:
    """Finds the time of day

    Args:
        hour (_type_): The time of day
    Returns:
        _type_: The time of day description string
    """
    if hour > 21 or hour <= 4:
        return 'Night'
    elif hour > 17:
        return 'Evening'
    elif hour > 12:
        return 'Afternoon'
    elif hour == 12:
        return 'Noon'
    elif hour > 4:
        return 'Morning'


def transform_features(df: pd.DataFrame, features: list, transformer: object, fit: bool) -> pd.DataFrame:
    """Transform features of the given dataframe

    Args:
        df (pd.DataFrame): The dataframe to be transformed
        features (list): The features of the dataframe to be transformed
        operator (object): The transformer to make the transformation
        fit (bool): Weather to run a fit before transform

    Returns:
        pd.DataFrame: _description_
    """

    original_index = df.index.name
    df = df.reset_index()

    transformed = transformer.fit_transform(
        df[features]) if fit else transformer.transform(df[features])

    transformed_df = pd.DataFrame(
        data=transformed, columns=transformer.get_feature_names_out())

    assert len(df) == len(
        transformed_df), 'The length of the dataframe changed after "transform"'

    df = df.drop(
        features, axis=1)

    joined_scaled_df = pd.merge(
        df, transformed_df, left_index=True, right_index=True)

    joined_scaled_df = joined_scaled_df.set_index(original_index)

    return joined_scaled_df


def is_rush_hour(x: pd.Series) -> bool:
    # print(x)
    """Checks weather the given hour and day would be a rush hour

    Args:
        x (_type_): The dataframe

    Returns:
        _type_: Returns True or False representing if it was a rush hour
    """
    if x['hour'] in [8, 17, 18] and x['day_of_week'] not in ['Saturday', 'Sunday']:
        return 1
    else:
        return 0


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a Time indexed dataframe with columns with the datetime information extracted

    Args:
        df (pd.DataFrame): The time series dataframe    

    Returns:
        pd.DataFrame: The existing dataframe with new columns
    """
    df['year'] = df.index.year
    df['month'] = df.index.month_name()
    df['day_of_week'] = df.index.day_name()
    df['hour'] = df.index.hour
    df['cos_hour'] = np.cos(2 * math.pi * df['hour'] / df['hour'].max())
    df['hour_sin'] = np.sin(2 * math.pi * df['hour']/24)
    
    df['month_sin'] = np.sin(2 * math.pi  * df.index.month/12)
    df['month_cos'] = np.cos(2 * math.pi  * df.index.month/12)


    df['is_weekend'] = ((df['day_of_week'] == 'Saturday') |
                        (df['day_of_week'] == 'Sunday')).astype(int)
    df['time_of_day'] = df['hour'].map(find_time_of_day)
    df['is_rush_hour'] = df.apply(is_rush_hour, axis=1)
    df['interaction_work_rush'] = df['workingday'] * df['is_rush_hour']

    # df['time_index'] = int(df.index.now().timestamp())

    return df


def engineer_df(df: pd.DataFrame, nm_fts: list, cat_fts: list, num_scaler: MinMaxScaler, poly: PolynomialFeatures, ohe: OneHotEncoder, fit: bool, rm=[]) -> pd.DataFrame:
    """Engineers an entire dataframe with the given features and transformers

    Args:
        df (pd.DataFrame): The dataframe to be transformed
        nm_fts (list): The numerical features
        cat_fts (list): The categorical features
        num_scaler (MinMaxScaler): The num scaler
        poly (PolynomialFeatures): The poly encoder
        ohe (OneHotEncoder): The Onehot encoder
        fit (bool): Weather to fit_transform or just transform.
        rm (list): Columns to be removed from the dataframe at the end of the engineering process

    Returns:
        pd.DataFrame: _description_
    """
    df = add_time_columns(df)

    df = transform_features(
        df, nm_fts, num_scaler, fit=fit)

    df = transform_features(
        df, nm_fts, poly, fit=fit)

    df = transform_features(
        df, cat_fts, ohe, fit=fit)

    if rm is not None:
        df.drop(rm, axis=1, inplace=True)

    return df
