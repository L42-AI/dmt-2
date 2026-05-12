import pandas as pd
import numpy as np
import lightgbm as lgb

__all__ = ["LambdaMARTRanker"]


class LambdaMARTRanker:
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

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Extract feature columns, excluding metadata and target columns."""
        exclude_cols = {
            'srch_id', 'prop_id', 'position', 'relevance',
            'date_time', 'checkin_date', 'click_bool', 'booking_bool',
            'gross_bookings_usd',
            'persona_latent_x', 'persona_latent_y'
        }
        return [col for col in df.columns if col not in exclude_cols]

    def _create_group_data(self, df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
        """Create group data for LightGBM ranking."""
        grouped = df.groupby('srch_id').size().values
        return grouped

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        """Train the LambdaMART model.

        Args:
            train_df: Training dataframe with 'srch_id' (group), features, and 'relevance' (target).
            val_df: Optional validation dataframe for early stopping.
        """
        # Get feature columns
        self.feature_names = self._get_feature_columns(train_df)
        X_train = train_df[self.feature_names].to_numpy(copy=False)
        y_train = train_df['relevance'].to_numpy(copy=False)
        groups_train = self._create_group_data(train_df)

        # Prepare LightGBM dataset
        train_data = lgb.Dataset(
            X_train, label=y_train, group=groups_train,
            feature_name=self.feature_names
        )

        # Prepare validation dataset if provided
        valid_data = None
        if val_df is not None:
            X_val = val_df[self.feature_names].to_numpy(copy=False)
            y_val = val_df['relevance'].to_numpy(copy=False)
            groups_val = self._create_group_data(val_df)

            valid_data = lgb.Dataset(
                X_val, label=y_val, group=groups_val,
                reference=train_data, feature_name=self.feature_names
            )

        # Train the model
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

        print(f"LambdaMART training complete. Trained on {len(train_df)} samples across {len(groups_train)} queries.")

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Generate ranking positions using the trained model.

        Args:
            test_df: Test dataframe with features and 'srch_id'.

        Returns:
            Dataframe with assigned 'position' column.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        df = test_df
        X_test = df[self.feature_names].to_numpy(copy=False)

        # Get prediction scores
        scores = self.model.predict(X_test)
        df['lambdamart_score'] = scores
        # Sort by srch_id and score (descending) to get ranking
        df.sort_values(by=['srch_id', 'lambdamart_score'], ascending=[True, False], inplace=True)

        # Assign positions within each search query
        df['position'] = df.groupby('srch_id').cumcount() + 1

        # Clean up temporary column
        df.drop(columns=['lambdamart_score'], inplace=True)
        return df
