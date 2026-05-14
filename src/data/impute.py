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

def process_competitor_variables(df: pd.DataFrame) -> pd.DataFrame:
    """ Process the competitor variables by removing negative values, and creating new variables to denote missingness."""
    for i in range(1, 9):
        rate_col = f'comp{i}_rate'
        inv_col = f'comp{i}_inv'
        diff_col = f'comp{i}_rate_percent_diff'

        missing_col = f'comp{i}_missing'
        has_better_price_col = f'comp{i}_has_better_price'
        has_worse_price_col = f'comp{i}_has_worse_price'

        df[missing_col] = (df[rate_col].isna() | df[inv_col].isna() | df[diff_col].isna()).astype('uint8')
        df[has_better_price_col] = (df[rate_col] < 0.0).fillna(False).astype('uint8')
        df[has_worse_price_col] = (df[rate_col] > 0.0).fillna(False).astype('uint8')

        df[inv_col] = df[inv_col].where(df[missing_col] == 0, other=-1.0)
        df[diff_col] = df[diff_col].where(df[missing_col] == 0, other=-1.0)

        df.drop(columns=[rate_col], inplace=True)

    return df