import pandas as pd

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

def build_missing_flags(df: pd.DataFrame, variable: str, value: float = 0.0) -> pd.DataFrame:
    """ Replaces specified values in variable with NA, and creates a new variable that 
    denotes when NA has occured for a hotel.

    Args: 
        df (pd.DataFrame)   : The dataframe to edit
        variable (string)   : The variable to check for specified value
        value (float)       : The value to replace with pd.NA
    """
    df[variable] = df[variable].replace(value, pd.NA)
    df[f'mflag_{variable}'] = df[variable].notna().astype(int)
    return df

def build_relevance_scores(df: pd.DataFrame) -> pd.DataFrame:
    ''' Build the relevance score attribute in the dataframe, according to the values specified in 
        the assignment document

    Args:
        df (pd.DataFrame): The dataframe to which to add the relevance score attribute
    '''
    clicked = df['click_bool'] == 1
    booked = df['booking_bool'] == 1
    neither = ~(clicked | booked)

    df.loc[clicked, 'rel_score'] = 1
    df.loc[booked, 'rel_score'] = 5
    df.loc[neither, 'rel_score'] = 0
    
    return df

def process_features(data, config):
    # Dummy implementation for feature engineering and imputation
    return data
