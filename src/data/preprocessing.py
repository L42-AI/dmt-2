import pandas as pd


from .split import train_val_split
from .scale import Scaler

from .features.ids import resample_ids
from .features.distance import GraphDistanceImputer
from .features.competitors import process_competitor_variables
from .features.features import convert_target_to_relevance_scores
from .features.date_time import add_date_features, build_checkin_dates, build_checkout_dates, add_weekend_proportion, build_binary_season

from .features.features import *

def _clean_impute_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    # === Prices     
    # Some prices are zero, which is impossible, need to be imputed
    df['price_usd'] = df['price_usd'].replace(0.0, np.nan)
    
    # Set maximum allowed price
    # max_logical_price = 5000.0 * df['srch_length_of_stay'] * df['srch_room_count']
    max_logical_price = 800

    # Properties above price-threshold are clipped to maximum price allowed
    absurd_price_mask = (df['price_usd'] <= 0.0) | (df['price_usd'] > max_logical_price)
    df.loc[absurd_price_mask, 'price_usd'] = max_logical_price

    # Set zero-valued prices and nans to median price of query (best guess)
    query_price_median = df.groupby('srch_id')['price_usd'].transform('median')
    df['price_usd'] = df['price_usd'].fillna(query_price_median)

    # Convert to cents
    df['price_usd'] = (df['price_usd'] * 100).round()
    
    # === Historical user star-rating
    dest_star_median = df.groupby('srch_destination_id')['prop_starrating'].transform('median')
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(dest_star_median)

    # === Historical user mean-price per night
    dest_price_median = df.groupby('srch_destination_id')['price_usd'].transform('median')
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(dest_price_median)
    df['visitor_hist_adr_usd_missing'] = df['visitor_hist_adr_usd'].isna().astype('uint8')

    # === Property star-rating
    # According to description, 0s are essentially missing values. No NAs in raw data.
    df['prop_starrating'] = df['prop_starrating'].replace(0.0, np.nan)
    df['prop_starrating_missing'] = df['prop_starrating'].isna().astype('uint8')

    # === Property review-scores
    # Imputation: Create two flags, no prior revies, and missing. Then impute the non-zero missing values. Finally set 0s to missing.
    df['prop_review_score_no_prior_reviews'] = df['prop_review_score'].where(df['prop_review_score'] == 0.0).notna().astype('uint8')
    df['prop_review_score_missing'] = df['prop_review_score'].isna().astype('uint8')
    # Get median reviews of hotels with same stars 
    star_review_median = df.groupby('prop_starrating', dropna=False)['prop_review_score'].transform('median')
    # Get median reviews of hotels with same location
    dest_review_median = df.groupby('srch_destination_id')['prop_review_score'].transform('median')
    # First try imputation based on hotels with similar stars
    df['prop_review_score'] = df['prop_review_score'].fillna(star_review_median)
    # Then try imputation based on destinations
    df['prop_review_score'] = df['prop_review_score'].fillna(dest_review_median)
    # Replace 0 with nans, which will be passed to the models.
    df['prop_review_score'] = df['prop_review_score'].replace(0.0, np.nan) # replace 0s so that scaling is unbiased

    # === Location scores
    # Score 1
    # TODO: Scaling
    df['prop_location_score1_missing'] = df['prop_location_score1'].isna().astype('uint8')
    dest_median = df.groupby('srch_destination_id')['prop_location_score1'].transform('median')
    df['prop_location_score1'] = df['prop_location_score1'].fillna(dest_median)
    
    # Fallbacks for score 1
    star_median = df.groupby('prop_starrating', dropna=False)['prop_location_score1'].transform('median')
    df['prop_location_score1'] = df['prop_location_score1'].fillna(star_median)
    df['prop_location_score1'] = df['prop_location_score1'].fillna(df['prop_location_score1'].median())

    # Score 2
    # TODO: Scaling
    df['prop_location_score2_missing'] = df['prop_location_score2'].isna().astype('uint8')
    dest_median2 = df.groupby('srch_destination_id')['prop_location_score2'].transform('median')
    df['prop_location_score2'] = df['prop_location_score2'].fillna(dest_median2)

    # Fallbacks
    star_median2 = df.groupby('prop_starrating', dropna=False)['prop_location_score2'].transform('median')
    df['prop_location_score2'] = df['prop_location_score2'].fillna(star_median2)
    df['prop_location_score2'] = df['prop_location_score2'].fillna(df['prop_location_score2'].median())

    # Interaction of location scores and disagreement
    df['location_score_interaction'] = df['prop_location_score1'] * df['prop_location_score2']
    df['location_score_interaction_missing'] = df['location_score_interaction'].isna().astype('uint8')

    normalized_score1 = df['prop_location_score1'] / df['prop_location_score1'].max()
    df['location_disagreement'] = df['prop_location_score2'] - normalized_score1

    # === Property historical price
    # TODO: Scaling
    # Clip negative prices
    df['prop_log_historical_price'] = df['prop_log_historical_price'].where(df['prop_log_historical_price'] >= 0.0)
    # Zeros have meaning, encode in binary feature
    df['prop_no_historical_price'] = (df['prop_log_historical_price'] == 0.0).astype('uint8')
    # Encode missing values in binary feature
    df['prop_log_historical_price_missing'] = df['prop_log_historical_price'].isna().astype('uint8')
    
    # === Expedia search affinity score
    # Note: Computed by expedia, we cannot guess this, and thus cannot impute. 
    # TODO: Scaling
    df['srch_query_affinity_score_missing'] = df['srch_query_affinity_score'].isna().astype('uint8')

    # TODO: Scaling
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
    # Dan: Currently tested
    df = build_urgency_variables(df)
    df = add_stats_per_prop(df)

    df = add_price_features(df)
    df = add_history_match_features(df)
    df = add_query_relative_features(df)
    df = process_competitor_variables(df)

    df['properties_per_query'] = df.groupby('srch_id')['prop_id'].transform('count')  

    df['is_domestic'] = (df['visitor_location_country_id'] == df['prop_country_id']).where(df['visitor_location_country_id'].notnull() & df['prop_country_id'].notnull(), 0).astype('uint8')
    return df

def select_randomized_instances(df: pd.DataFrame) -> pd.DataFrame:
    return df.query('random_bool == 1')

def preprocess_data(train_set: pd.DataFrame, test_set: pd.DataFrame) -> tuple:

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

    categoricals = ['srch_id', 'site_id', 'prop_country_id', 'prop_id', 'srch_destination_id', 
                    'prop_brand_bool', 'srch_saturday_night_bool', 'random_bool']
    scaler = Scaler(exclude = categoricals + ['date_time', 'checkin_date', 'checkout_date', 'relevance'])
    train_set, val_set, test_set = scaler.fit_transform(train_set, val_set, test_set)

    return train_set, val_set, test_set
