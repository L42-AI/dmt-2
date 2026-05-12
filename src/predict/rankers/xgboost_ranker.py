import pandas as pd
from xgboost import XGBRanker

from .rank_data_preprocessor import RankDataPreprocessor

__all__ = ["XGBoostRanker"]


class XGBoostRanker(RankDataPreprocessor):
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
        train_df = train_df.sort_values(by='srch_id', ascending=True)

        X_train = train_df[self.feature_names].to_numpy(copy=False)
        y_train = train_df['relevance'].to_numpy(copy=False)
        group_train = self._create_group_data(train_df)

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
            val_df = val_df.sort_values(by='srch_id', ascending=True)
            X_val = val_df[self.feature_names].to_numpy(copy=False)
            y_val = val_df['relevance'].to_numpy(copy=False)
            group_val = self._create_group_data(val_df)
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['eval_group'] = [group_val]

        self.model.fit(**fit_kwargs)
        print(f"XGBoost training complete. Trained on {len(train_df)} rows across {train_df['srch_id'].nunique()} queries.")

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        df = test_df
        X_test = df[self.feature_names].to_numpy(copy=False)
        df['xgboost_score'] = self.model.predict(X_test)
        df.sort_values(by=['srch_id', 'xgboost_score'], ascending=[True, False], inplace=True)
        df['position'] = df.groupby('srch_id').cumcount() + 1
        df.drop(columns=['xgboost_score'], inplace=True)
        return df
