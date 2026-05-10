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

def create_relevance_target(df: pd.DataFrame) -> pd.DataFrame:
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