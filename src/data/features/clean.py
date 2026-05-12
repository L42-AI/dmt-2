import pandas as pd

def remove_negative(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """ Remove negative values from a variable column in a dataframe

    Args:
        df (pd.DataFrame)   : The dataframe to edit
        variable (string)   : The variable column to remove the negative values from
    """

    mask = df[variable] < 0
    df.loc[mask, variable] = pd.NA
    return df

def set_missing_to_zero(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """ Set values in a variable column to zero

    Args:
        df (pd.DataFrame)   : The dataframe to edit
        variable (string)   : The variable column to set to zero
    """

    df[variable] = df[variable].fillna(0)
    return df
