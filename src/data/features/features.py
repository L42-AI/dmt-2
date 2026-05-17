import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PropertyStatsTransformer(BaseEstimator, TransformerMixin):
    """
    Computes property-level statistics from train and maps them onto all splits.
    Runs after Imputer so it operates on clean, consistently-scaled columns.
    """

    COLS = [
        'price_usd',
        'prop_starrating',
        'prop_review_score',
        'prop_location_score1',
        'prop_location_score2',
    ]
    CTR_CVR_GROUPS = ['prop_starrating'] 
    CTR_CVR_BINS   = [('price_usd', 50), ('location_score_interaction', 50)]

    def __init__(self):
        self.stats_          = {}
        self.global_means_   = {}
        self.global_medians_ = {}
        self.global_stds_    = {}
        self.ctr_ = {}
        self.cvr_ = {}

    def fit(self, df: pd.DataFrame, y=None) -> "PropertyStatsTransformer":
        for col in self.COLS:
            if col not in df.columns:
                continue
            self.stats_[col] = {
                'mean':   df.groupby('prop_id')[col].mean(),
                'std':    df.groupby('prop_id')[col].std(),
                'median': df.groupby('prop_id')[col].median(),
            }
            self.global_means_[col]   = df[col].mean()
            self.global_medians_[col] = df[col].median()  # correct fallback for median columns
            self.global_stds_[col]    = df[col].std()

        clicked = df[df['relevance'] > 0]
        self.global_click_rate_   = (df['relevance'] > 0).mean()
        self.global_booking_rate_ = (df['relevance'] == 5).mean()

        for group in self.CTR_CVR_GROUPS:
            presentations = df.groupby(group).size()
            clicks        = (df['relevance'] > 0).groupby(df[group]).sum()
            bookings      = (clicked['relevance'] == 5).groupby(clicked[group]).sum()
            self.ctr_[group] = clicks / presentations
            self.cvr_[group] = bookings / clicks.replace(0, np.nan)

        self.bin_edges_ = {}
        for col, n_bins in self.CTR_CVR_BINS:
            _, edges = pd.qcut(df[col], q=n_bins, retbins=True, duplicates='drop')
            edges[0], edges[-1] = -np.inf, np.inf
            self.bin_edges_[col] = edges

            bucket = pd.cut(df[col], bins=edges, labels=False)
            clicked_bucket = pd.cut(clicked[col], bins=edges, labels=False)
            presentations = df.groupby(bucket).size()
            clicks        = (df['relevance'] > 0).groupby(bucket).sum()
            bookings      = (clicked['relevance'] == 5).groupby(clicked_bucket).sum()
            self.ctr_[col] = clicks / presentations
            self.cvr_[col] = bookings / clicks.replace(0, np.nan)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, stat_maps in self.stats_.items():
            if col not in df.columns:
                continue

            mean_col   = f'prop_mean_{col}'
            std_col    = f'prop_std_{col}'
            median_col = f'prop_median_{col}'

            df[mean_col]   = df['prop_id'].map(stat_maps['mean']).fillna(self.global_means_[col])
            df[median_col] = df['prop_id'].map(stat_maps['median']).fillna(self.global_medians_[col])
            df[std_col]    = df['prop_id'].map(stat_maps['std']).fillna(0)

            df[f'{col}_minus_prop_mean'] = df[col] - df[mean_col]
            df[f'{col}_div_prop_mean'] = (
                df[col] / df[mean_col].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)

            std_fallback = self.global_stds_.get(col, 1.0) or 1.0
            std_series = df[std_col].replace(0, np.nan).fillna(std_fallback)
            df[f'{col}_prop_zscore'] = (
                (df[col] - df[mean_col]) / std_series
            ).replace([np.inf, -np.inf], np.nan)
        for group in self.CTR_CVR_GROUPS:
            df[f'{group}_ctr'] = df[group].map(self.ctr_[group]).fillna(self.global_click_rate_)
            df[f'{group}_cvr'] = df[group].map(self.cvr_[group]).fillna(self.global_booking_rate_)
        for col, _ in self.CTR_CVR_BINS:
            bucket = pd.cut(df[col], bins=self.bin_edges_[col], labels=False)
            df[f'{col}_bucket_ctr'] = bucket.map(self.ctr_[col]).fillna(self.global_click_rate_)
            df[f'{col}_bucket_cvr'] = bucket.map(self.cvr_[col]).fillna(self.global_booking_rate_)
        return df


def convert_target_to_relevance_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Maps click/booking booleans to the competition relevance scale (0 / 1 / 5)."""
    df['relevance'] = np.select(
        [df['booking_bool'] == 1, df['click_bool'] == 1],
        [5, 1],
        default=0,
    )
    df.drop(columns=['click_bool', 'booking_bool', 'gross_bookings_usd', 'position'],
            inplace=True)
    return df


def add_travel_party_features(df: pd.DataFrame) -> pd.DataFrame:
    df['srch_total_guests']  = df['srch_adults_count'] + df['srch_children_count']
    df['srch_has_children']  = (df['srch_children_count'] > 0).astype('uint8')
    df['srch_is_family']     = (
        (df['srch_adults_count'] >= 2) & (df['srch_children_count'] > 0)
    ).astype('uint8')
    df['srch_is_solo']       = (
        (df['srch_adults_count'] == 1) & (df['srch_children_count'] == 0)
    ).astype('uint8')
    df['srch_guests_per_room'] = df['srch_total_guests'] / df['srch_room_count'].clip(lower=1)
    return df


def build_urgency_variables(df: pd.DataFrame) -> pd.DataFrame:
    df['last_minute']        = (df['srch_booking_window'] < 7).astype('uint8')
    df['price_urgency_ratio'] = df['price_usd'] / (df['srch_booking_window'] + 1)
    if 'promotion_flag' in df.columns:
        df['last_minute_promo'] = df['last_minute'] * df['promotion_flag']
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """All price values are in cents at this stage."""
    df['price_usd_log'] = np.log1p(df['price_usd'])

    nights = df['srch_length_of_stay'].clip(lower=1)
    guests = df['srch_total_guests'].clip(lower=1) if 'srch_total_guests' in df.columns else 1

    df['price_per_night']      = df['price_usd'] / nights
    df['price_per_guest']      = df['price_usd'] / guests
    df['price_per_guest_night'] = df['price_usd'] / (guests * nights)
    df['price_per_star']       = df['price_usd'] / df['prop_starrating'].replace(0, np.nan)
    df['price_per_review_score'] = df['price_usd'] / df['prop_review_score'].replace(0, np.nan)
    return df


def add_history_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares current hotel to the user's historical preferences.
    price_usd is in cents; visitor_hist_adr_usd is in dollars.
    Dividing by 100 aligns units before computing the ratio.
    """
    df['star_vs_user_history'] = df['prop_starrating'] - df['visitor_hist_starrating']

    price_dollars = df['price_usd'] / 100
    hist_adr      = df['visitor_hist_adr_usd']
    df['price_vs_user_hist_adr'] = (
        (price_dollars - hist_adr) / hist_adr.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    return df


def add_query_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within-search ranking features. Columns absent in a given split are skipped.
    test_random_star / test_random_location are dataset-specific and may only
    appear in the test split.
    """
    cols = [
        'price_usd', 'price_usd_log', 'price_per_night',
        'prop_starrating', 'prop_review_score',
        'prop_location_score1', 'prop_location_score2',
        'prop_log_historical_price', 'srch_query_affinity_score',
        'orig_destination_distance',
    ]

    new_cols = {}
    for col in cols:
        if col not in df.columns:
            continue
        x   = df[col]
        grp = x.groupby(df['srch_id'])

        mean    = grp.transform('mean')
        std     = grp.transform('std').replace(0, np.nan)
        min_val = grp.transform('min')
        max_val = grp.transform('max')

        new_cols[f'{col}_minus_query_mean']   = x - mean
        new_cols[f'{col}_div_query_mean']     = (x / mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        new_cols[f'{col}_query_zscore']       = ((x - mean) / std).replace([np.inf, -np.inf], np.nan)
        new_cols[f'{col}_minus_query_median'] = x - grp.transform('median')
        new_cols[f'{col}_query_pct_rank']     = grp.rank(pct=True)
        new_cols[f'{col}_is_query_min']       = (x == min_val).astype('uint8')
        new_cols[f'{col}_is_query_max']       = (x == max_val).astype('uint8')

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df