from .features.scale import scale_scores
from .features.features import *
import pandas as pd

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

def _clean_impute_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].where(df['visitor_hist_starrating'] >= 1.0)
    df['visitor_hist_starrating'] = scale_scores(df, 'visitor_hist_starrating')
    df['visitor_hist_starrating_missing'] = df['visitor_hist_starrating'].isna().astype('uint8')

    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].where(df['visitor_hist_adr_usd'] > 0.0)
    df['visitor_hist_adr_usd_missing'] = df['visitor_hist_adr_usd'].isna().astype('uint8')
    
    df['prop_starrating'] = df['prop_starrating'].where(df['prop_starrating'] >= 1.0)
    df['prop_starrating'] = scale_scores(df, 'prop_starrating')
    df['prop_starrating_missing'] = df['prop_starrating'].isna().astype('uint8')

    df['prop_review_score_no_prior_reviews'] = df['prop_review_score'].where(df['prop_review_score'] == 0.0).notna().astype('uint8')
    df['prop_review_score'] = df['prop_review_score'].where(df['prop_review_score'] >= 1.0)
    df['prop_review_score'] = scale_scores(df, 'prop_review_score')
    df['prop_review_score_missing'] = (df['prop_review_score'].isna() & df['prop_review_score_no_prior_reviews'] == 0).astype('uint8')

    # TODO: Scaling
    df['prop_location_score1'] = df['prop_location_score1'].where(df['prop_location_score1'] > 0.0)
    df['prop_location_score1_missing'] = df['prop_location_score1'].isna().astype('uint8')

    # TODO: Scaling
    df['prop_location_score2'] = df['prop_location_score2'].where(df['prop_location_score2'] > 0.0)
    df['prop_location_score2_missing'] = df['prop_location_score2'].isna().astype('uint8')

    # TODO: Scaling
    df['prop_log_historical_price'] = df['prop_log_historical_price'].where(df['prop_log_historical_price'] > 0.0)
    df['prop_log_historical_price_missing'] = df['prop_log_historical_price'].isna().astype('uint8')

    # Convert to cents
    df['price_usd'] = (df['price_usd'] * 100).round().astype('int32')

    # TODO: srch_query_affinity_score
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].apply(lambda x : np.exp(x) if x is not pd.NA else x)
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].where(df['prop_log_historical_price'] < 0.0)
    df['srch_query_affinity_score_missing'] = df['srch_query_affinity_score'].isna().astype('uint8')
    
    # TODO: orig_destination_distance
    df['orig_destination_distance'] = df['orig_destination_distance'].where(df['prop_log_historical_price'] > 0.0)
    df['orig_destination_distance_missing'] = df['orig_destination_distance'].isna().astype('uint8')

    df = impute_missing_distances(df)

    return df

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


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Build new features from existing ones, for example by combining them or applying transformations."""
    df['checkin_date'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'].squeeze(), unit='D') # type: ignore[arg-type]
    df['checkout_date'] = df['checkin_date'] + pd.to_timedelta(df['srch_length_of_stay'].squeeze(), unit='D') # type: ignore[arg-type]
    df = build_binary_season(name = 'off_season', df = df, start = 8, end = 11, ref='date_time')
    df = build_prop_avg_price(df)
    df = add_date_features(df)
    df = add_travel_party_features(df)
    df = add_price_features(df)
    df = add_history_match_features(df)
    df = add_competitor_summary_features(df)
    df = add_query_relative_features(df)
    df = process_competitor_variables(df)
    df = add_distance_bucketization(df)
    df = add_weekend_proportion(df)

    df['properties_per_query'] = df.groupby('srch_id')['prop_id'].transform('count')  

    df['is_domestic'] = (df['visitor_location_country_id'] == df['prop_country_id']).where(df['visitor_location_country_id'].notnull() & df['prop_country_id'].notnull(), 0).astype('uint8')
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Clean, Impute and Scale existing features
    df = _clean_impute_and_scale(df)
    
    # 2. Build new features
    df = _engineer_features(df)
    return df
