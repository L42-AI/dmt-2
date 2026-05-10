import pandas as pd
import numpy as np

def compute_comp_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the competition rate for each search.
    """
    for num in range(1, 9):
        percent_diff = pd.to_numeric(df[f'comp{num}_rate_percent_diff'], errors='coerce')
        price_usd = pd.to_numeric(df['price_usd'], errors='coerce')
        df[f'comp{num}_rate'] = percent_diff * price_usd

    return df

def build_relevance_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps clicks and bookings to the Kaggle competition relevance scale.
    Assumes your training data has 'click_bool' and 'booking_bool' columns.
    This is for building the target variable during training, not for test set predictions.
    """
    conditions = [
        df['booking_bool'] == 1,
        df['click_bool'] == 1
    ]
    choices = [5, 1]
    
    # 5 if booked, 1 if clicked (but not booked), 0 otherwise
    df['relevance'] = np.select(conditions, choices, default=0)

    return df.drop(columns=['click_bool', 'booking_bool', 'gross_bookings_usd', 'position'])

def build_binary_season(name: str, df: pd.DataFrame, start: int, end: int, ref: str = 'date_time') -> pd.DataFrame:
    """Build a boolean variable which flags whether a date-time of a query is within a specified
    period. For example, if one can define a high-season, one can check whether the query 
    happended within that season. The start and end boundaries are inclusive. 

    Args:
        name (string):      The name of the new variable
        df (pd.DataFrame):  The dataframe to edit
        start (int):        The starting month of the specified period
        end (int):          The last month of the specified period
        ref (string):       The variable to reference. For example query datetime, 
                            or check-in date. 
    """
    df[name] = df[ref].dt.month.between(start, end) # .between is inclusuve !
    return df

def build_checkin_dates(df: pd.DataFrame) -> pd.DataFrame:
    """ Builds a variable which determines the intended check-in date in a search query 
    from 'date_time' and 'srch_booking_window'. 

    Args:
        df (pd.DataFrame)   : The dataframe to edit
    """
    df['checkin_date'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'].squeeze(), unit='D') # type: ignore[arg-type]
    return df

def build_flag_variable(name: str, df: pd.DataFrame, variable: str, value: float = 0.0) -> pd.DataFrame:
    """ Replaces specified values in variable with NA, and creates a new variable that 
    denotes when NA has occured for a hotel.

    Args: 
        df (pd.DataFrame)   : The dataframe to edit
        variable (string)   : The variable to check for specified value
        value (float)       : The value to replace with pd.NA
    """
    # First, create flag variable
    df[name] = df[variable][df[variable] == value].astype(int)
    # Then, replace the specified values with NAs
    df[variable] = df[variable].replace(value, pd.NA)
    return df


