import pandas as pd


from .split import train_val_split
from .scale import scale_scores

from .features.ids import resample_ids
from .features.distance import GraphDistanceImputer
from .features.competitors import process_competitor_variables
from .features.features import convert_target_to_relevance_scores
from .features.date_time import add_date_features, build_checkin_dates, build_checkout_dates, add_weekend_proportion, build_binary_season

from .features.features import *

def _clean_impute_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].where(df['visitor_hist_starrating'] >= 1.0)
    df['visitor_hist_starrating_missing'] = df['visitor_hist_starrating'].isna().astype('uint8')

    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].where(df['visitor_hist_adr_usd'] > 0.0)
    df['visitor_hist_adr_usd_missing'] = df['visitor_hist_adr_usd'].isna().astype('uint8')
    
    df['prop_starrating'] = df['prop_starrating'].where(df['prop_starrating'] >= 1.0)
    df['prop_starrating_missing'] = df['prop_starrating'].isna().astype('uint8')

    df['prop_review_score_no_prior_reviews'] = df['prop_review_score'].where(df['prop_review_score'] == 0.0).notna().astype('uint8')
    df['prop_review_score'] = df['prop_review_score'].where(df['prop_review_score'] >= 1.0)
    df['prop_review_score'] = scale_scores(df, 'prop_review_score')
    df['prop_review_score_missing'] = (df['prop_review_score'].isna() & df['prop_review_score_no_prior_reviews'] == 0).astype('uint8')

    # Convert to cents
    df['price_usd'] = (df['price_usd'] * 100).round().astype('int32')

    # Impute missing star history with the median STAR RATING of the destination
    dest_star_median = df.groupby('srch_destination_id')['prop_starrating'].transform('median')
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(dest_star_median)
    df['visitor_hist_starrating'] = scale_scores(df, 'visitor_hist_starrating') # Scale AFTER filling
    df['prop_starrating'] = scale_scores(df, 'prop_starrating') # Scale AFTER filling

    # Impute missing ADR (budget) with the median HOTEL PRICE of the destination
    dest_price_median = df.groupby('srch_destination_id')['price_usd'].transform('median')
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(dest_price_median)

    # TODO: Scaling
    df['prop_location_score1'] = df['prop_location_score1'].where(df['prop_location_score1'] > 0.0)
    df['prop_location_score1_missing'] = df['prop_location_score1'].isna().astype('uint8')

    # TODO: Scaling
    df['prop_location_score2'] = df['prop_location_score2'].where(df['prop_location_score2'] > 0.0)
    df['prop_location_score2_missing'] = df['prop_location_score2'].isna().astype('uint8')

    df['location_score_interaction'] = df['prop_location_score1'] * df['prop_location_score2']
    df['location_score_interaction_missing'] = df['location_score_interaction'].isna().astype('uint8')

    normalized_score1 = df['prop_location_score1'] / df['prop_location_score1'].max()
    df['location_disagreement'] = df['prop_location_score2'] - normalized_score1

    # TODO: Scaling
    df['prop_log_historical_price'] = df['prop_log_historical_price'].where(df['prop_log_historical_price'] > 0.0)
    df['prop_log_historical_price_missing'] = df['prop_log_historical_price'].isna().astype('uint8')

    # TODO: srch_query_affinity_score
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].apply(lambda x : np.exp(x) if x is not pd.NA else x)
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].where(df['srch_query_affinity_score'] >= 0.0)
    df['srch_query_affinity_score_missing'] = df['srch_query_affinity_score'].isna().astype('uint8')
    
    # TODO: orig_destination_distance
    df['orig_destination_distance'] = df['orig_destination_distance'].where(df['orig_destination_distance'] > 0.0)
    df['orig_destination_distance_missing'] = df['orig_destination_distance'].isna().astype('uint8')

    return df

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Build new features from existing ones, for example by combining them or applying transformations."""
    
    # Build date features
    df = build_checkin_dates(df)
    df = build_checkout_dates(df)
    df = add_date_features(df)
    df = add_weekend_proportion(df)
    df = build_binary_season(name = 'off_season', df = df, start = 8, end = 11, ref='date_time')

    df = build_prop_avg_price(df)
    df = add_travel_party_features(df)
    df = add_price_features(df)
    df = add_history_match_features(df)
    df = add_query_relative_features(df)
    df = process_competitor_variables(df)

    df['properties_per_query'] = df.groupby('srch_id')['prop_id'].transform('count')  

    df['is_domestic'] = (df['visitor_location_country_id'] == df['prop_country_id']).where(df['visitor_location_country_id'].notnull() & df['prop_country_id'].notnull(), 0).astype('uint8')
    return df

def preprocess_data(train_set: pd.DataFrame, test_set: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    complete_set = pd.concat([train_set, test_set], ignore_index=True)
    complete_set = resample_ids(complete_set)
    train_set, test_set = complete_set.iloc[:len(train_set)], complete_set.iloc[len(train_set):]
    del complete_set

    train_set = convert_target_to_relevance_scores(train_set)

    train_set, val_set = train_val_split(train_set, 0.8)

    graph_imputer = GraphDistanceImputer()
    graph_imputer.fit(train_set)
    
    train_set = graph_imputer.transform(train_set)
    val_set = graph_imputer.transform(val_set)
    test_set = graph_imputer.transform(test_set)

    train_set = _clean_impute_and_scale(train_set)
    val_set = _clean_impute_and_scale(val_set)
    test_set = _clean_impute_and_scale(test_set)
    
    train_set = _engineer_features(train_set)
    val_set = _engineer_features(val_set)
    test_set = _engineer_features(test_set)

    return train_set, val_set, test_set
