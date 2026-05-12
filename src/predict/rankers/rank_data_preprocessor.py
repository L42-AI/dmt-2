import numpy as np
import pandas as pd

__all__ = ["RankDataPreprocessor"]


class RankDataPreprocessor:
    """Shared preprocessing helpers for learning-to-rank models."""

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