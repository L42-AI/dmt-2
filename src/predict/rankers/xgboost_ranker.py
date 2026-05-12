import pandas as pd
from xgboost import XGBRanker

from .rank_data_processor import RankDataProcessor

__all__ = ["XGBoostRanker"]


class XGBoostRanker(RankDataProcessor):
    """XGBoost ranking model for hotel search ranking."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.feature_names = None

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        self.feature_names = self._get_feature_columns(train_df)
        train_df, X_train, y_train, group_train = self._prepare_rank_data(train_df, self.feature_names)

        self.model = XGBRanker(
            objective='rank:ndcg',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            tree_method='hist',
            eval_metric='ndcg@5',
        )

        fit_kwargs = {
            'X': X_train,
            'y': y_train,
            'group': group_train,
            'verbose': False,
        }

        if val_df is not None:
            val_df, X_val, y_val, group_val = self._prepare_rank_data(val_df, self.feature_names)
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['eval_group'] = [group_val]

        self.model.fit(**fit_kwargs)
        print(f"XGBoost training complete. Trained on {len(train_df)} rows across {train_df['srch_id'].nunique()} queries.")

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        scores = self.model.predict(test_df[self.feature_names].to_numpy(copy=False))
        return self._apply_rank_scores(test_df, scores)
