import pandas as pd

def impute_missing_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use other known distance data to impute missing distances
    """
    # Get known distances median
    dist_v_dest = df.groupby(['visitor_location_country_id', 'srch_destination_id'])['orig_destination_distance'].transform('median')
    dist_v_prop = df.groupby(['visitor_location_country_id', 'prop_country_id'])['orig_destination_distance'].transform('median')
    
    # fill with median
    df['imputed_distance'] = df['orig_destination_distance']
    df['imputed_distance'] = df['imputed_distance'].fillna(dist_v_dest)
    df['imputed_distance'] = df['imputed_distance'].fillna(dist_v_prop)

    # Calculate distance z_scores
    std = df.groupby('srch_id')['imputed_distance'].transform('std').replace(0, 1)
    df['distance_z_score'] = (df['imputed_distance'] - df.groupby('srch_id')['imputed_distance'].transform('mean')) / std
    
    return df