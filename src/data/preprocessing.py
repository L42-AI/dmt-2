from .features.scale import scale_scores
from .features.features import build_binary_season
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
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(-1)

    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].where(df['visitor_hist_adr_usd'] > 0.0)
    df['visitor_hist_adr_usd_missing'] = df['visitor_hist_adr_usd'].isna().astype('uint8')
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(-1)
    
    df['prop_starrating'] = df['prop_starrating'].where(df['prop_starrating'] >= 1.0)
    df['prop_starrating'] = scale_scores(df, 'prop_starrating')
    df['prop_starrating_missing'] = df['prop_starrating'].isna().astype('uint8')
    df['prop_starrating'] = df['prop_starrating'].fillna(-1)

    df['prop_review_score_no_prior_reviews'] = df['prop_review_score'].where(df['prop_review_score'] == 0.0).notna().astype('uint8')
    df['prop_review_score'] = df['prop_review_score'].where(df['prop_review_score'] >= 1.0)
    df['prop_review_score'] = scale_scores(df, 'prop_review_score')
    df['prop_review_score_missing'] = (df['prop_review_score'].isna() & df['prop_review_score_no_prior_reviews'] == 0).astype('uint8')
    df['prop_review_score'] = df['prop_review_score'].fillna(-1)

    # TODO: Scaling
    df['prop_location_score1'] = df['prop_location_score1'].where(df['prop_location_score1'] > 0.0)
    df['prop_location_score1_missing'] = df['prop_location_score1'].isna().astype('uint8')
    df['prop_location_score1'] = df['prop_location_score1'].fillna(-1)

    # TODO: Scaling
    df['prop_location_score2'] = df['prop_location_score2'].where(df['prop_location_score2'] > 0.0)
    df['prop_location_score2_missing'] = df['prop_location_score2'].isna().astype('uint8')
    df['prop_location_score2'] = df['prop_location_score2'].fillna(-1)

    # TODO: Scaling
    df['prop_log_historical_price'] = df['prop_log_historical_price'].where(df['prop_log_historical_price'] > 0.0)
    df['prop_log_historical_price_missing'] = df['prop_log_historical_price'].isna().astype('uint8')
    df['prop_log_historical_price'] = df['prop_log_historical_price'].fillna(-1)

    # Convert to cents
    df['price_usd'] = (df['price_usd'] * 100).round().astype('int32')

    # TODO: srch_query_affinity_score
    # NOTE: Flip to positive numbers to have everything on positive range, IMPORTANT SEMANTIC CHOICE
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].apply(lambda x : -x if x is not pd.NA else x)
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].where(df['prop_log_historical_price'] > 0.0)
    df['srch_query_affinity_score_missing'] = df['srch_query_affinity_score'].isna().astype('uint8')
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(-1)
    
    # TODO: orig_destination_distance
    df['orig_destination_distance'] = df['orig_destination_distance'].where(df['prop_log_historical_price'] > 0.0)
    df['orig_destination_distance_missing'] = df['orig_destination_distance'].isna().astype('uint8')
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(-1)

    df = process_competitor_variables(df)

    return df

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Build new features from existing ones, for example by combining them or applying transformations."""
    df['checkin_date'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'].squeeze(), unit='D') # type: ignore[arg-type]
    df = build_binary_season(name = 'off_season', df = df, start = 8, end = 11, ref='date_time')
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Clean, Impute and Scale existing features
    df = _clean_impute_and_scale(df)
    
    # 2. Build new features
    df = _engineer_features(df)
    return df
