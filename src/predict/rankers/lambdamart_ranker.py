import pandas as pd
import lightgbm as lgb

from .rank_data_processor import RankDataProcessor

__all__ = ["LambdaMARTRanker"]


class LambdaMARTRanker(RankDataProcessor):
    """LambdaMART learning-to-rank model for hotel search ranking."""

    def __init__(self, num_leaves: int = 31, learning_rate: float = 0.1, n_estimators: int = 100):
        """Initialize the LambdaMART ranker.

        Args:
            num_leaves: Number of leaves in each tree.
            learning_rate: Learning rate for boosting.
            n_estimators: Number of boosting rounds.
        """
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.model = None
        self.feature_names = None

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Train the LambdaMART model.

        Args:
            train_df: Training dataframe with 'srch_id' (group), features, and 'relevance' (target).
            val_df: Optional validation dataframe for early stopping.
        """
        self.feature_names = self._get_feature_columns(train_df)
        train_df, X_train, y_train, group_train = self._prepare_rank_data(train_df, self.feature_names)

        train_data = lgb.Dataset(
            X_train, label=y_train, group=group_train,
            feature_name=self.feature_names
        )

        valid_data = None
        if val_df is not None:
            val_df, X_val, y_val, group_val = self._prepare_rank_data(val_df, self.feature_names)

            valid_data = lgb.Dataset(
                X_val, label=y_val, group=group_val,
                reference=train_data, feature_name=self.feature_names
            )

        params = {
            'objective': 'rank_xendcg',
            'metric': 'ndcg',
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'verbose': -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_data] if valid_data is not None else None,
            callbacks=[lgb.early_stopping(10)] if valid_data is not None else None
        )

        print(f"LambdaMART training complete. Trained on {len(train_df)} samples across {len(group_train)} queries.")

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
