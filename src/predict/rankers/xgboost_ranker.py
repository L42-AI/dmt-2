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
        subsample: float = 1.0,           
        colsample_bytree: float = 1.0,    
        min_child_weight: int = 1,        
        gamma: float = 0.0,            
        reg_lambda: float = 1.0,       
        random_state: int = 42,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma             
        self.reg_lambda = reg_lambda   
        self.random_state = random_state

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        self.feature_names = self._get_feature_columns(train_df)
        train_df, X_train, y_train, group_train = self._prepare_rank_data(train_df, self.feature_names)

        if val_df is not None:
            val_df, X_val, y_val, group_val = self._prepare_rank_data(val_df, self.feature_names)

        self.model = XGBRanker(
            objective='rank:ndcg',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,               
            reg_lambda=self.reg_lambda,     
            random_state=self.random_state,
            tree_method='hist',
            eval_metric='ndcg@5',
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
