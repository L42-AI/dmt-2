import numpy as np
import pandas as pd


class Imputer:
    """Fits global imputation statistics on train, applies to all splits."""

    def fit(self, df: pd.DataFrame) -> "Imputer":
        # Exclude structural zeros before computing fallback statistics
        self.global_price_median_    = df['price_usd'].replace(0.0, np.nan).median()
        self.price_cap_              = df['price_usd'].replace(0.0, np.nan).quantile(0.995)
        self.prop_starrating_median_ = df['prop_starrating'].replace(0.0, np.nan).median()
        self.prop_review_global_med_ = df['prop_review_score'].replace(0.0, np.nan).median()
        self.loc1_global_med_        = df['prop_location_score1'].median()
        self.loc2_global_med_        = df['prop_location_score2'].median()
        self.dist_median_            = df['orig_destination_distance'].median()
        self.loc1_max_               = df['prop_location_score1'].max()
        # ADR is in dollars — fitted separately so it is never confused with price_usd
        self.global_adr_median_      = df['visitor_hist_adr_usd'].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Visitor ADR history ---
        # Must be imputed BEFORE price_usd is converted to cents below
        df['visitor_hist_adr_usd_missing'] = df['visitor_hist_adr_usd'].isna().astype('uint8')
        dest_adr_median = df.groupby('srch_destination_id')['visitor_hist_adr_usd'].transform('median')
        df['visitor_hist_adr_usd'] = (df['visitor_hist_adr_usd']
            .fillna(dest_adr_median)
            .fillna(self.global_adr_median_))

        # --- Price ---
        df['price_usd'] = df['price_usd'].replace(0.0, np.nan)
        query_price_median = df.groupby('srch_id')['price_usd'].transform('median')
        df['price_usd'] = (df['price_usd']
            .fillna(query_price_median)
            .fillna(self.global_price_median_)
            .clip(upper=self.price_cap_))
        df['price_usd'] = (df['price_usd'] * 100).round()  # cents from here on

        # --- Property star-rating ---
        df['prop_starrating'] = df['prop_starrating'].replace(0.0, np.nan)
        df['prop_starrating_missing'] = df['prop_starrating'].isna().astype('uint8')
        df['prop_starrating'] = df['prop_starrating'].fillna(self.prop_starrating_median_)

        # --- Visitor star-rating history ---
        dest_star_median = df.groupby('srch_destination_id')['prop_starrating'].transform('median')
        df['visitor_hist_starrating'] = (df['visitor_hist_starrating']
            .fillna(dest_star_median)
            .fillna(self.prop_starrating_median_))

        # --- Review score ---
        df['prop_review_score_no_prior_reviews'] = (
            df['prop_review_score'].where(df['prop_review_score'] == 0.0).notna().astype('uint8')
        )
        df['prop_review_score_missing'] = df['prop_review_score'].isna().astype('uint8')
        star_review_median = df.groupby('prop_starrating', dropna=False)['prop_review_score'].transform('median')
        dest_review_median = df.groupby('srch_destination_id')['prop_review_score'].transform('median')
        df['prop_review_score'] = (df['prop_review_score']
            .fillna(star_review_median)
            .fillna(dest_review_median)
            .replace(0.0, np.nan)
            .fillna(self.prop_review_global_med_))

        # --- Location score 1 ---
        df['prop_location_score1_missing'] = df['prop_location_score1'].isna().astype('uint8')
        dest_median = df.groupby('srch_destination_id')['prop_location_score1'].transform('median')
        star_median = df.groupby('prop_starrating', dropna=False)['prop_location_score1'].transform('median')
        df['prop_location_score1'] = (df['prop_location_score1']
            .fillna(dest_median).fillna(star_median).fillna(self.loc1_global_med_))

        # --- Location score 2 ---
        df['prop_location_score2_missing'] = df['prop_location_score2'].isna().astype('uint8')
        dest_median2 = df.groupby('srch_destination_id')['prop_location_score2'].transform('median')
        star_median2 = df.groupby('prop_starrating', dropna=False)['prop_location_score2'].transform('median')
        df['prop_location_score2'] = (df['prop_location_score2']
            .fillna(dest_median2).fillna(star_median2).fillna(self.loc2_global_med_))

        # --- Distance ---
        df['orig_destination_distance'] = (df['orig_destination_distance']
            .fillna(self.dist_median_)
            .where(lambda s: s > 0.0))

        # --- Interaction & disagreement features ---
        df['location_score_interaction'] = df['prop_location_score1'] * df['prop_location_score2']
        df['location_score_interaction_missing'] = df['location_score_interaction'].isna().astype('uint8')
        df['location_disagreement'] = (
            df['prop_location_score2'] - (df['prop_location_score1'] / self.loc1_max_)
        )

        # --- Missingness flags for columns without imputation ---
        df['prop_log_historical_price'] = df['prop_log_historical_price'].where(
            df['prop_log_historical_price'] >= 0.0)
        df['prop_no_historical_price'] = (df['prop_log_historical_price'] == 0.0).astype('uint8')
        df['prop_log_historical_price_missing'] = df['prop_log_historical_price'].isna().astype('uint8')
        df['srch_query_affinity_score_missing'] = df['srch_query_affinity_score'].isna().astype('uint8')

        # --- Logit transform of location score 2 ---
        # Applied here so PropertyStatsTransformer (next stage) learns stats
        # on the same scale the model will see.
        eps = 1e-6
        df['prop_location_score2'] = np.log(
            (df['prop_location_score2'] + eps) / (1 - df['prop_location_score2'] + eps)
        )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)