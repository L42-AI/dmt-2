
import pandas as pd
import numpy as np

def build_checkin_dates(df: pd.DataFrame) -> pd.DataFrame:
    """ Builds a variable which determines the intended check-in date in a search query 
    from 'date_time' and 'srch_booking_window'. 

    Args:
        df (pd.DataFrame)   : The dataframe to edit
    """
    df['checkin_date'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'].squeeze(), unit='D') # type: ignore[arg-type]
    return df

def build_checkout_dates(df: pd.DataFrame) -> pd.DataFrame:
    """ Builds a variable which determines the intended check-out date in a search query 
    from 'checkin_date' and 'srch_length_of_stay'. 

    Args:
        df (pd.DataFrame)   : The dataframe to edit
    """
    df['checkout_date'] = df['checkin_date'] + pd.to_timedelta(df['srch_length_of_stay'].squeeze(), unit='D') # type: ignore[arg-type]
    return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract model-usable date features from date_time and checkin_date
    """
    df['search_month'] = df['date_time'].dt.month.astype('uint8') # Months are 1-12, so uint8 is sufficient
    df['search_dayofweek'] = df['date_time'].dt.dayofweek.astype('uint8') # Monday=0, Sunday=6
    df['search_hour'] = df['date_time'].dt.hour.astype('uint8') # 0-23, so uint8 is sufficient

    if 'checkin_date' in df.columns:
        df['checkin_month'] = df['checkin_date'].dt.month.astype('uint8') 
        df['checkin_dayofweek'] = df['checkin_date'].dt.dayofweek.astype('uint8')
        df['checkin_weekend'] = df['checkin_dayofweek'].isin([5, 6]).astype('uint8')

    if 'checkout_date' in df.columns:
        df['checkout_month'] = df['checkout_date'].dt.month.astype('uint8') 
        df['checkout_dayofweek'] = df['checkout_date'].dt.dayofweek.astype('uint8')
        df['checkout_weekend'] = df['checkout_dayofweek'].isin([5, 6]).astype('uint8')

    return df

def add_weekend_proportion(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Calculates the proportion of the stay that falls on a weekend 
    using fast numpy vectorization instead of row-by-row apply.
    """
    df = df.copy()
    
    # Ensure they are proper datetime types at the day resolution
    checkin = df['checkin_date'].values.astype('datetime64[D]')
    checkout = df['checkout_date'].values.astype('datetime64[D]')
    
    # 1. Calculate total stay length in days
    total_days = (checkout - checkin).astype('timedelta64[D]').astype(int)
    
    # 2. Count the business days between the dates 
    # (np.busday_count automatically uses a 'left' inclusive logic by default)
    business_days = np.busday_count(checkin, checkout)
    
    # 3. Weekend days are just the difference
    weekend_days = total_days - business_days
    
    # 4. Calculate proportion safely (avoiding divide-by-zero if someone booked 0 days)
    df['weekend_proportion'] = np.where(
        total_days > 0, 
        weekend_days / total_days, 
        0.0
    )
    
    return df

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