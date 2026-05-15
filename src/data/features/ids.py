import pandas as pd

def resample_ids(df: pd.DataFrame):
    """
    Resample:
    - 'srch_destination_id'
    - 'visitor_location_country_id'
    - 'prop_country_id'
    """

    # Resample 'srch_destination_id'
    unique_destinations = df['srch_destination_id'].drop_duplicates()
    destination_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_destinations)}
    df['srch_destination_id'] = df['srch_destination_id'].map(destination_mapping)

    # Resample 'visitor_location_country_id' and 'prop_country_id' together
    unique_countries = pd.concat([df['visitor_location_country_id'], df['prop_country_id']]).drop_duplicates()
    country_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_countries)}
    df['visitor_location_country_id'] = df['visitor_location_country_id'].map(country_mapping)
    df['prop_country_id'] = df['prop_country_id'].map(country_mapping)
    return df