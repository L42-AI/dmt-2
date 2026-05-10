import pandas as pd

def train_val_split(data: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.sort_values('date_time')

    sessions = data[['srch_id', 'date_time']].drop_duplicates()

    cutoff_index = int(len(sessions) * train_ratio)
    cutoff_date = sessions.iloc[cutoff_index]['date_time']

    train_data = data[data['date_time'] <= cutoff_date]
    test_data = data[data['date_time'] > cutoff_date]

    return train_data, test_data