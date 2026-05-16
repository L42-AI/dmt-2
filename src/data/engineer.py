import pandas as pd

from .features.date_time import (
    build_checkin_dates,
    build_checkout_dates,
    add_date_features,
    add_weekend_proportion,
    build_binary_season,
)
from .features.competitors import process_competitor_variables
from .features.features import (
    add_travel_party_features,
    build_urgency_variables,
    add_price_features,
    add_history_match_features,
    add_query_relative_features,
)


class FeatureEngineer:
    """
    Stateless feature engineering pipeline.

    All transforms here are either pure arithmetic or within-query aggregations —
    neither requires fitting on training data. The fit/transform interface is kept
    for consistency with Imputer and PropertyStatsTransformer, and to allow future
    stateful additions without changing the calling code in preprocessing.py.

    Ordering constraints (enforced by call order in transform):
        1. Checkin/checkout dates before any date features that consume them.
        2. Travel party features before price features (price_per_guest needs srch_total_guests).
        3. All base features before query-relative features.
    """

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Date features ---
        df = build_checkin_dates(df)
        df = build_checkout_dates(df)
        df = add_date_features(df)
        df = add_weekend_proportion(df)
        df = build_binary_season(name='off_season', df=df, start=8, end=11, ref='date_time')

        # --- Travel party (before price features) ---
        df = add_travel_party_features(df)

        # --- Urgency signals ---
        df = build_urgency_variables(df)

        # --- Price & history features ---
        df = add_price_features(df)
        df = add_history_match_features(df)

        # --- Competitor features ---
        df = process_competitor_variables(df)

        # --- Query-relative features (must be last: uses all base columns) ---
        df = add_query_relative_features(df)

        # --- Simple derived ---
        df['properties_per_query'] = df.groupby('srch_id')['prop_id'].transform('count')
        df['is_domestic'] = (
            df['visitor_location_country_id'] == df['prop_country_id']
        ).where(
            df['visitor_location_country_id'].notnull() & df['prop_country_id'].notnull(),
            other=0,
        ).astype('uint8')

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)