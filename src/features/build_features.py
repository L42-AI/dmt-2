import pandas as pd

def build_relevance_scores(df: pd.DataFrame) -> pd.DataFrame:
    ''' Build the relevance score attribute in the dataframe, according to the values specified in 
        the assignment document

    Args:
        df (pd.DataFrame): The dataframe to which to add the relevance score attribute
    '''
    which_clicked = df.query('click_bool == 1').index
    which_booked = df.query('booking_bool == 1').index
    which_neither = df.query('click_bool == 0 & booking_bool == 0')
    df.loc[which_clicked, 'rel_score'] = 1
    df.loc[which_booked, 'rel_score'] = 5
    df.loc[which_neither, 'rel_score'] = 0
    
    return df


def process_features(data, config):
    # Dummy implementation for feature engineering and imputation
    return data
