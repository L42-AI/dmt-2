import pandas as pd

def resample_ids(df: pd.DataFrame):
    """
    Resample:
    - 'srch_destination_id'
    - 'visitor_location_country_id'
    - 'prop_country_id'
    """
    for col in ['srch_destination_id', 'visitor_location_country_id', 'prop_country_id']:
        unique_values = df[col].unique()
        mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_values)}
        df[col] = df[col].map(mapping).astype('int32')
    return df