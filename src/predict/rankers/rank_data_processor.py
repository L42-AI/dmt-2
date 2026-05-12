import numpy as np
import pandas as pd

import lightgbm as lgb
from xgboost import XGBRanker

__all__ = ["RankDataProcessor"]


class RankDataProcessor:
    """Shared preprocessing helpers for learning-to-rank models."""

    model: XGBRanker | lgb.Booster | None
    feature_names: list[str] | None

    def __init__(self):
        self.model = None
        self.feature_names = None

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Extract feature columns, excluding metadata, target, and ranking columns."""
        exclude_cols = {
            'srch_id',
            'prop_id',
            'relevance',
            'date_time',
            'checkin_date',
            'position',
        }
        return [col for col in df.columns if col not in exclude_cols]

    def _create_group_data(self, df: pd.DataFrame) -> np.ndarray:
        """Create per-query group sizes for ranking objectives."""
        return df.groupby('srch_id').size().to_numpy()

    def _sort_for_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort rows so query groups are contiguous before ranking fit."""
        return df.sort_values(by=['srch_id'], ascending=[True])

    def _prepare_rank_data(self, df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Sort a dataframe and extract the arrays needed by ranking models."""
        ranked_df = self._sort_for_ranking(df)
        features = ranked_df[feature_names].to_numpy(copy=False)
        labels = ranked_df['relevance'].to_numpy(copy=False)
        groups = self._create_group_data(ranked_df)
        return ranked_df, features, labels, groups

    def _apply_rank_scores(self, df: pd.DataFrame, scores: np.ndarray, score_column: str = 'rank_score') -> pd.DataFrame:
        """Attach scores, sort within query, and assign 1-based positions."""
        ranked_df = df.copy()
        ranked_df[score_column] = scores
        ranked_df.sort_values(by=['srch_id', score_column], ascending=[True, False], inplace=True)
        ranked_df['position'] = ranked_df.groupby('srch_id').cumcount() + 1
        ranked_df.drop(columns=[score_column], inplace=True)
        return ranked_df
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Generate ranking positions using the trained model.

        Args:
            test_df: Test dataframe with features and 'srch_id'.

        Returns:
            Dataframe with assigned 'position' column.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        scores = self.model.predict(test_df[self.feature_names].to_numpy(copy=False))
        return self._apply_rank_scores(test_df, scores)