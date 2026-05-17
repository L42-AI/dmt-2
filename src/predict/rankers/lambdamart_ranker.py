import pandas as pd
import lightgbm as lgb

from .rank_data_processor import RankDataProcessor

__all__ = ["LambdaMARTRanker"]


class LambdaMARTRanker(RankDataProcessor):
    """LambdaMART learning-to-rank model for hotel search ranking."""

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, feature_list: list[str] | None = None) -> None:
        """Train the LambdaMART model.

        Args:
            train_df: Training dataframe with 'srch_id' (group), features, and 'relevance' (target).
            val_df: Optional validation dataframe for early stopping.
        """
        # Use an explicit feature list when provided to ensure consistency
        if feature_list is not None:
            self.feature_names = feature_list
        else:
            self.feature_names = self._get_feature_columns(train_df)

        train_df, X_train, y_train, group_train = self._prepare_rank_data(train_df, self.feature_names)
        train_position = train_df['position'].values if 'position' in train_df.columns else None

        train_data = lgb.Dataset(
            X_train, label=y_train, group=group_train,
            feature_name=self.feature_names,
            position=train_position
        )

        valid_data = None
        if val_df is not None:
            val_df, X_val, y_val, group_val = self._prepare_rank_data(val_df, self.feature_names)
            val_position = val_df['position'].values if 'position' in val_df.columns else None

            valid_data = lgb.Dataset(
                X_val, label=y_val, group=group_val,
                reference=train_data, feature_name=self.feature_names,
                position=val_position
            )

        params = self.params
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 100),
            valid_sets=[valid_data] if valid_data is not None else None,
            callbacks=[lgb.early_stopping(50)] if valid_data is not None else None
        )

        print(f"LambdaMART training complete. Trained on {len(train_df)} samples across {len(group_train)} queries.")

    def get_feature_importance(self, importance_type = 'split', ascending: bool = False) -> pd.DataFrame:
        importance_values = self.model.feature_importance(importance_type = importance_type)
        feature_names = self.model.feature_name()
        importance_df = pd.DataFrame({
            'feature' : feature_names,
            'importance': importance_values
        })
        importance_df = importance_df.sort_values(by='importance', ascending=ascending).reset_index(drop=True)
        return importance_df

