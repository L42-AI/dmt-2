import pandas as pd

def build_binary_season(name: str, df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    df[name] = df['date_time'].dt.month.between(start, end)
    return df

def build_checkin_dates(df: pd.DataFrame) -> pd.DataFrame:
    df['checkin_date'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'].squeeze(), unit='D') # type: ignore[arg-type]
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
