import pandas as pd
import lightgbm as lgb

from .rank_data_processor import RankDataProcessor

__all__ = ["LambdaMARTRanker"]


class LambdaMARTRanker(RankDataProcessor):
    """LambdaMART learning-to-rank model for hotel search ranking."""

    def __init__(
        self,
        parameters: dict,
        random_state: int = 42,
    ):
        super().__init__()
        self.parameters = parameters
        self.random_state = random_state
        
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
            'eval_at': [5],
            'num_leaves': self.parameters.get('num_leaves'),
            'max_depth': self.parameters.get('max_depth'),
            'min_child_samples': self.parameters.get('min_child_samples'),
            "learning_rate": self.parameters.get('learning_rate'),
            "subsample": self.parameters.get('subsample'),
            "colsample_bytree": self.parameters.get('colsample_bytree'),
            "min_gain_to_split": self.parameters.get('min_gain_to_split'),
            "reg_alpha": self.parameters.get('reg_alpha'),
            "reg_lambda": self.parameters.get('reg_lambda'),
            
            'random_state': self.random_state,
            'verbose': -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.parameters.get('n_estimators'),
            valid_sets=[valid_data] if valid_data is not None else None,
            callbacks=[lgb.early_stopping(10)] if valid_data is not None else None
        )

        print(f"LambdaMART training complete. Trained on {len(train_df)} samples across {len(group_train)} queries.")
