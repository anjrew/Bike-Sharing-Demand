import numpy as np
import pandas as pd
from statistics import mode


def has_mode(arr: np.array) -> bool: 
    """Checks if mode is available with the given array

    Args:
        arr (np.array): The array to check

    Returns:
        bool: A boolean representing if Mode is available
    """
    return len(set(arr.tolist())) != len(arr.tolist()) 


def get_mode_or_mean(ds_in: pd.Series) -> pd.DataFrame:
    """Gets the mode or mean if the are available

    Args:
        ds_in (pd.Series): The data series in

    Returns:
        pd.DataFrame: The data Series back
    """
    count = None
    if(has_mode(ds_in.to_numpy())):
        try: 
            count = mode(ds_in.to_numpy())
        except:
            count = np.around(np.mean(ds_in.to_numpy()))
    else:
        count = np.around(np.mean(ds_in.to_numpy()))
    ds_in['count'] = count
    return ds_in