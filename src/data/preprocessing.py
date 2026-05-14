from .features.scale import scale_scores
#[R] from .features.features import add_query_relative_features, build_binary_season, add_date_features, add_travel_party_features, add_price_features, add_history_match_features, add_competitor_summary_features
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

    # === Price in usd: This variable is incredibly vague. Conversion rates to usd and night or whole stay cannot be ascertained.
    # Some prices are zero, which is impossible, need to be imputed
    df['price_usd'] = df['price_usd'].replace(0.0, np.nan)

    # Assuming a sane person is maximally willing to pay 5000 usd per stay. Calculate the maximum possible price per stay.
    # Also assuming the price is per room, who knows? 
    max_logical_price = 5000.0 * df['srch_length_of_stay'] * df['srch_room_count']
    absurd_price_mask = (df['price_usd'] <= 0.0) | (df['price_usd'] > max_logical_price)
    df.loc[absurd_price_mask, 'price_usd'] = np.nan

    # At least the prices in the search query must be the same format. So use the query median as best guess.
    query_price_median = df.groupby('srch_id')['price_usd'].transform('median')
    df['price_usd'] = df['price_usd'].fillna(query_price_median)
    
    # == Historical user star-rating
    # Note: Imputation here does not make sense, as we cannot guess the rating of someone who never used Expedia
    df['visitor_hist_starrating'] = scale_scores(df, 'visitor_hist_starrating')
    df['visitor_hist_starrating_missing'] = df['visitor_hist_starrating'].isna().astype('uint8')

    # == Historical user mean-price per night
    # Note: No imputation, same logic as above.
    # TODO: Scaling
    df['visitor_hist_adr_usd_missing'] = df['visitor_hist_adr_usd'].isna().astype('uint8')

    # === Property star-rating
    # Imputation: According to description, 0s are essentially missing values. No NAs in raw data.
    df['prop_starrating'] = df['prop_starrating'].replace(0.0, np.nan)
    df['prop_starrating'] = scale_scores(df, 'prop_starrating')
    df['prop_starrating_missing'] = df['prop_starrating'].isna().astype('uint8')

    # === Property review-scores
    # Imputation: Create two flags, no prior revies, and missing. Then impute the non-zero missing values. Finally set 0s to missing.
    # Imputation 2: I choose to impute based on other hotels with similar star-rating. Reasoning is that hotels with 
    #               similar stars probably get similar reviews. Second choice is based on destinations.
    df['prop_review_score_no_prior_reviews'] = df['prop_review_score'].where(df['prop_review_score'] == 0.0).notna().astype('uint8')
    df['prop_review_score_missing'] = df['prop_review_score'].isna().astype('uint8')

    star_review_median = df.groupby('prop_starrating', dropna=False)['prop_review_score'].transform('median')
    dest_review_median = df.groupby('srch_destination_id')['prop_review_score'].transform('median')

    df['prop_review_score'] = df['prop_review_score'].fillna(star_review_median)
    df['prop_review_score'] = df['prop_review_score'].fillna(dest_review_median)

    df['prop_review_score'] = df['prop_review_score'].replace(0.0, np.nan) # replace 0s so that scaling is unbiased
    df['prop_review_score'] = scale_scores(df, 'prop_review_score')

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

    # === Property historical price
    # Note: No imputation, zero value has meaning, just create binary flag and replace with nans. 
    # TODO: Scaling
    df['prop_no_historical_price'] = (df['prop_log_historical_price'] == 0.0).astype('uint8')
    df['prop_log_historical_price'] = df['prop_log_historical_price'].replace(0.0, np.nan)

    # Convert to cents
    # df['price_usd'] = (df['price_usd'] * 100).round()

    # === Expedia search affinity score
    # Note: Computed by expedia, we cannot guess this, and thus cannot impute. 
    # NOTE: Flip to positive numbers to have everything on positive range, IMPORTANT SEMANTIC CHOICE
    # TODO: Scaling
    # df['srch_query_affinity_score'] = df['srch_query_affinity_score'].apply(lambda x : -x if x is not pd.NA else x)
    df['srch_query_affinity_score_missing'] = df['srch_query_affinity_score'].isna().astype('uint8')

    # === Distance from user location to hotel location 
    # TODO: Scaling
    df['orig_destination_distance_missing'] = df['orig_destination_distance'].isna().astype('uint8')
    
    # Imputation: Median distance between specific country pairs
    country_pair_median = df.groupby(['visitor_location_country_id', 'prop_country_id'])['orig_destination_distance'].transform('median')
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(country_pair_median)

    # Fallback imputation: Median distance to the destination country
    prop_country_median = df.groupby('prop_country_id')['orig_destination_distance'].transform('median')
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(prop_country_median)

    df = process_competitor_variables(df)

    return df

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Build new features from existing ones, for example by combining them or applying transformations."""
    df['checkin_date'] = df['date_time'] + pd.to_timedelta(df['srch_booking_window'].squeeze(), unit='D') # type: ignore[arg-type]
    # df = build_binary_season(name = 'off_season', df = df, start = 8, end = 11, ref='date_time')
    df = build_prop_avg_price(df)
    df = build_urgency_variables(df)
    df = build_test(df)
    # df = build_five_star_flag(df)
    df = add_stats_per_prop(df)

    df = add_date_features(df)
    #df = add_travel_party_features(df)
    df = add_price_features(df)
    df = add_history_match_features(df)
    df = add_competitor_summary_features(df)
    df = add_query_relative_features(df)
    
    return df

def select_randomized_instances(df: pd.DataFrame) -> pd.DataFrame:
    return df.query('random_bool == 1')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Clean, Impute and Scale existing features
    df = _clean_impute_and_scale(df)
    
    # 2. Build new features
    df = _engineer_features(df)
    return df


def cross_set_preprocessing(original_train_set: pd.DataFrame, train_set: pd.DataFrame, validation_set: pd.DataFrame, test_set: pd.DataFrame):
    """ Preprocessing using exclusive variables from the training set. This is a seperate function because the exclusive variables are 
    removed early in the pipeline, and the following steps are difficult to implement in the default preprocessing function. 
    This function ASSUMES that no instances have been deleted in preprocessing, otherwise there will be a mismatch in fitting.

    Args:
        original_train_set (pd.DataFrame): _description_
        train_set (pd.DataFrame): _description_
        validation_set (pd.DataFrame): _description_
        test_set (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    from lightgbm import LGBMRegressor

    # ctr and cvr per prop_id
    # global_ctr = original_train_set['click_bool'].sum() / len(original_train_set)
    # global_cvr = original_train_set['booking_bool'].sum() / original_train_set['click_bool'].sum()

    # agg = original_train_set.groupby('prop_id').agg(
    # presentations=('prop_id', 'count'),
    # clicks=('click_bool', 'sum'),
    # bookings=('booking_bool', 'sum')
    # )

    # alpha = 10  # smoothing strength
    # agg['ctr'] = (agg['clicks'] + alpha * global_ctr) / (agg['presentations'] + alpha)
    # agg['cvr'] = (agg['bookings'] + alpha * global_cvr) / (agg['clicks'] + alpha) 
    # for df in [train_set, validation_set, test_set]:
    #     df['ctr'] = df['prop_id'].map(agg['ctr'])
    #     df['cvr'] = df['prop_id'].map(agg['cvr'])
    
    # positions
    # features = [feat for feat in train_set.columns if feat not in ['position', 'booking_bool', 'click_bool', 'gross_bookings_usd', 'date_time', 'checkin_date', 'relevance']]
    # model = LGBMRegressor()
    # model.fit(
    #     train_set[features],
    #     original_train_set['position']
    # )
    # for df in [train_set, validation_set, test_set]:
    #     df['predicted_pos'] = model.predict(df[features])

    return train_set, validation_set, test_set