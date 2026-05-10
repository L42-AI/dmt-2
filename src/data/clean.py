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
