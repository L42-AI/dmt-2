import pandas as pd
from xgboost import XGBRanker

from .rank_data_processor import RankDataProcessor

__all__ = ["XGBoostRanker"]


class XGBoostRanker(RankDataProcessor):
    """XGBoost ranking model for hotel search ranking."""

    def __init__(
        self,
        parameters: dict,
        random_state: int = 42,
    ):
        super().__init__()
        self.parameters = parameters  
        self.random_state = random_state

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, feature_list: list[str] | None = None) -> None:
        # Use an explicit feature list when provided to ensure consistency
        if feature_list is not None:
            self.feature_names = feature_list
        else:
            self.feature_names = self._get_feature_columns(train_df)

        train_df, X_train, y_train, group_train = self._prepare_rank_data(train_df, self.feature_names)

        if val_df is not None:
            val_df, X_val, y_val, group_val = self._prepare_rank_data(val_df, self.feature_names)

        self.model = XGBRanker(
            objective='rank:ndcg',
            tree_method='hist',
            eval_metric='ndcg@5',
            grow_policy='lossguide',

            n_estimators=self.parameters.get('n_estimators'),
            max_leaves=self.parameters.get('max_leaves'),
            max_depth=self.parameters.get('max_depth'),
            min_child_weight=self.parameters.get('min_child_weight'),
            learning_rate=self.parameters.get('learning_rate'),
            subsample=self.parameters.get('subsample'),
            colsample_bytree=self.parameters.get('colsample_bytree'),
            colsample_bylevel=self.parameters.get('colsample_bylevel'),
            gamma=self.parameters.get('gamma'),
            reg_lambda=self.parameters.get('reg_lambda'),
            reg_alpha=self.parameters.get('reg_alpha'),
            
            random_state=self.random_state,
        )

        self.model.fit(
            X = X_train,
            y = y_train,
            group = group_train,
            verbose = False,
            eval_set = [(X_val, y_val)] if val_df is not None else None,
            eval_group = [group_val] if val_df is not None else None,
        )

        print(f"XGBoost training complete. Trained on {len(train_df)} rows across {train_df['srch_id'].nunique()} queries.")
