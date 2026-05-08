import pandas as pd

def train_val_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the input data on a ratio into train and test sets.
    Uses the date_time column to ensure a time-based split.
    Each srch_id should be entirely in either the train or test set to prevent data leakage.
    Each srch_id represents a unique search session, and has its own date_time, so we can use the date_time to split the data chronologically.
    """

    # Sort data by date_time to ensure chronological splitting
    data = data.sort_values('date_time')

    # Get unique search sessions (srch_id) and their corresponding date_time
    sessions = data[['srch_id', 'date_time']].drop_duplicates()

    # Determine the cutoff date for the train/test split based on the specified ratio
    cutoff_index = int(len(sessions) * train_ratio)
    cutoff_date = sessions.iloc[cutoff_index]['date_time']

    # Split the data into train and test sets based on the cutoff date
    train_data = data[data['date_time'] <= cutoff_date]
    test_data = data[data['date_time'] > cutoff_date]

    return train_data, test_data