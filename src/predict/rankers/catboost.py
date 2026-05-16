import pandas as pd
import numpy as np
from catboost import CatBoostRanker as CatBoostRankerModel, Pool

# Assuming RankDataProcessor is your base class based on your XGBoost implementation
class CatBoostRanker: 
    def __init__(
        self, 
        iterations: int = 500, 
        learning_rate: float = 0.05, 
        depth: int = 6, 
        l2_leaf_reg: float = 3.0,
        cat_features: list | None = None
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        # These are the high-cardinality IDs CatBoost loves
        self.cat_features = cat_features if cat_features else [
            'prop_id', 'srch_destination_id', 
            'prop_country_id', 'visitor_location_country_id'
        ]
        
        self.model = CatBoostRankerModel(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function='YetiRank', # Natively optimizes for NDCG
            eval_metric='NDCG:top=5',
            random_seed=42,
            verbose=100
        )
        self.feature_names = None

    def _prepare_pool(self, df: pd.DataFrame, is_train: bool = True) -> Pool:
        """Prepares the CatBoost Pool object and safely formats categorical columns."""
        df_clean = df.copy()
        
        # Sort by srch_id (CatBoost requires groups to be contiguous in the dataframe)
        df_clean = df_clean.sort_values('srch_id').reset_index(drop=True)
        
        # Safely cast categorical features to strings (CatBoost rejects float columns)
        for col in self.cat_features:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(-1).astype(int).astype(str)

        # Ensure we only pass existing categorical features to the Pool
        valid_cat_features = [col for col in self.cat_features if col in df_clean.columns]

        # Extract components
        group_id = df_clean['srch_id']

        # If feature_names provided, use them; otherwise drop known non-features
        if self.feature_names is not None:
            features = df_clean[self.feature_names].copy()
        else:
            features = df_clean.drop(columns=['srch_id', 'relevance', 'position', 'click_bool', 'booking_bool'], errors='ignore')

        target = df_clean['relevance'] if is_train else None

        return Pool(
            data=features,
            label=target,
            group_id=group_id,
            cat_features=valid_cat_features
        )

    def train(self, train_set: pd.DataFrame, val_set: pd.DataFrame) -> None:
        def _ensure_feature_names():
            # If feature_names not set, infer from train_set by excluding known metadata
            if self.feature_names is None:
                self.feature_names = [c for c in train_set.select_dtypes(include=['number', 'bool']).columns if c not in ['srch_id', 'relevance', 'position']]

        print("\n--- Preparing CatBoost Data Pools ---")
        _ensure_feature_names()
        train_pool = self._prepare_pool(train_set, is_train=True)
        val_pool = self._prepare_pool(val_set, is_train=True)

        print("--- Training CatBoost Ranker ---")
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            use_best_model=True
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        # Ensure we select same features as training
        pred_pool = self._prepare_pool(df, is_train=False)
        
        # Get raw continuous predictions
        df_out['raw_score'] = self.model.predict(pred_pool)
        
        # Rank them descending per search (higher raw score = better rank)
        df_out['position'] = df_out.groupby('srch_id')['raw_score'].rank(
            method='first', 
            ascending=False
        ).astype('int32')
        
        return df_out.drop(columns=['raw_score'])