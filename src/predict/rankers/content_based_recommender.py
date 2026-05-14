import numpy as np
import pandas as pd

from .rank_data_processor import RankDataProcessor


__all__ = ["ContentKnowledgeRecommender"]


class ContentKnowledgeRecommender(RankDataProcessor):
    """
    Content-/knowledge-based recommender baseline for hotel ranking.

    This is not a learned ranking model like LambdaMART or XGBoost.
    Instead, it builds an interpretable ranking score using:
      - hotel content: review score, location score, star rating, brand, promotion
      - search context: price relative to other hotels in the same query
      - user profile/history: match with historical star rating and ADR
      - simple popularity priors learned from the training data

    The final score is used to rank hotels within each srch_id.
    """

    def __init__(
        self,
        property_smoothing: float = 20.0,
        destination_smoothing: float = 10.0,
    ):
        super().__init__()
        self.property_smoothing = property_smoothing
        self.destination_smoothing = destination_smoothing

        self.global_relevance_mean: float | None = None
        self.property_prior: pd.DataFrame | None = None
        self.destination_property_prior: pd.DataFrame | None = None

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """
        Estimate simple popularity priors from the training set only.

        These priors are smoothed averages of the relevance score:
          relevance = 5 for booking, 1 for click, 0 otherwise.
        """
        if "relevance" not in train_df.columns:
            raise ValueError("train_df must contain a 'relevance' column.")

        self.global_relevance_mean = float(train_df["relevance"].mean())

        # Property-level prior: how often this hotel was clicked/booked historically.
        prop_stats = (
            train_df
            .groupby("prop_id")["relevance"]
            .agg(["mean", "count"])
            .reset_index()
        )

        prop_stats["property_prior_score"] = (
            (prop_stats["mean"] * prop_stats["count"])
            + (self.global_relevance_mean * self.property_smoothing)
        ) / (prop_stats["count"] + self.property_smoothing)

        self.property_prior = prop_stats[["prop_id", "property_prior_score"]]

        # Destination-property prior: how well this hotel performed in this destination.
        if "srch_destination_id" in train_df.columns:
            dest_prop_stats = (
                train_df
                .groupby(["srch_destination_id", "prop_id"])["relevance"]
                .agg(["mean", "count"])
                .reset_index()
            )

            dest_prop_stats["destination_property_prior_score"] = (
                (dest_prop_stats["mean"] * dest_prop_stats["count"])
                + (self.global_relevance_mean * self.destination_smoothing)
            ) / (dest_prop_stats["count"] + self.destination_smoothing)

            self.destination_property_prior = dest_prop_stats[
                ["srch_destination_id", "prop_id", "destination_property_prior_score"]
            ]

        print(
            "Content/knowledge-based recommender trained. "
            f"Global relevance mean: {self.global_relevance_mean:.4f}"
        )

    @staticmethod
    def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        Safe column getter.

        Missing columns get a constant default.
        Missing values and encoded -1 values are replaced by default.
        """
        if name not in df.columns:
            return pd.Series(default, index=df.index, dtype="float64")

        return (
            pd.to_numeric(df[name], errors="coerce")
            .replace(-1, np.nan)
            .fillna(default)
            .astype("float64")
        )

    @staticmethod
    def _match_score(x: pd.Series) -> pd.Series:
        """
        Convert an absolute difference into a 0-1 compatibility score.

        Difference close to 0 -> score close to 1.
        Large difference -> lower score.
        """
        return 1.0 / (1.0 + x.abs())

    def _add_priors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge popularity priors learned on the training set.
        """
        ranked_df = df.copy()

        fallback = self.global_relevance_mean if self.global_relevance_mean is not None else 0.0

        if self.property_prior is not None:
            ranked_df = ranked_df.merge(
                self.property_prior,
                on="prop_id",
                how="left",
            )
        else:
            ranked_df["property_prior_score"] = fallback

        ranked_df["property_prior_score"] = ranked_df["property_prior_score"].fillna(fallback)

        if (
            self.destination_property_prior is not None
            and "srch_destination_id" in ranked_df.columns
        ):
            ranked_df = ranked_df.merge(
                self.destination_property_prior,
                on=["srch_destination_id", "prop_id"],
                how="left",
            )
        else:
            ranked_df["destination_property_prior_score"] = fallback

        ranked_df["destination_property_prior_score"] = (
            ranked_df["destination_property_prior_score"]
            .fillna(ranked_df["property_prior_score"])
            .fillna(fallback)
        )

        return ranked_df

    def _score(self, df: pd.DataFrame) -> pd.Series:
        """
        Build an interpretable content-/knowledge-based score.

        Higher score = hotel should be ranked higher within the search query.
        """
        # Item/content quality.
        review = self._col(df, "prop_review_score_query_pct_rank", 0.5)
        location1 = self._col(df, "prop_location_score1_query_pct_rank", 0.5)
        location2 = self._col(df, "prop_location_score2_query_pct_rank", 0.5)
        star = self._col(df, "prop_starrating_query_pct_rank", 0.5)

        # Price/value. Since pct_rank is high for expensive hotels, invert it.
        cheap_price = 1.0 - self._col(df, "price_usd_query_pct_rank", 0.5)
        cheap_guest_night = 1.0 - self._col(df, "price_per_guest_night_query_pct_rank", 0.5)

        # Query affinity was flipped to positive in preprocessing, so lower is better.
        affinity = 1.0 - self._col(df, "srch_query_affinity_score_query_pct_rank", 0.5)

        # Hotel/business signals.
        promotion = self._col(df, "promotion_flag", 0.0)
        brand = self._col(df, "prop_brand_bool", 0.0)

        # Knowledge-based matching to user history.
        star_match = self._match_score(self._col(df, "star_vs_user_history", 0.0))
        price_match = self._match_score(self._col(df, "price_vs_user_hist_adr", 0.0))

        # Popularity priors from training data.
        property_prior = self._col(df, "property_prior_score", 0.0)
        destination_property_prior = self._col(df, "destination_property_prior_score", 0.0)

        # Optional family/search-party logic.
        # For family searches, value per guest-night matters slightly more.
        is_family = self._col(df, "srch_is_family", 0.0)
        family_value_bonus = is_family * cheap_guest_night

        score = (
            1.20   * destination_property_prior
            + 0.80 * property_prior
            + 0.90 * review
            + 0.80 * location1
            + 0.50 * location2
            + 0.50 * affinity
            + 0.45 * cheap_price
            + 0.35 * cheap_guest_night
            + 0.30 * star
            + 0.30 * star_match
            + 0.25 * price_match
            + 0.20 * promotion
            + 0.10 * brand
            + 0.20 * family_value_bonus
        )

        return score

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank hotels within each search query.
        """
        ranked_df = self._add_priors(test_df)
        scores = self._score(ranked_df)

        return self._apply_rank_scores(ranked_df, scores)