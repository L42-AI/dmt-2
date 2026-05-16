from typing import Literal
import pandas as pd

from data import load_data
from data import preprocess_data
from data.feature_selection import select_feature_columns
from predict import (
    clear_predictions,
    random,
    ceiling,
    CatBoostRanker,
    LambdaMARTRanker,
    XGBoostRanker,
    ContentKnowledgeRecommender
)
from evaluate import compute_accuracy


class Pipeline:

    TRAIN_RATIO = 0.8

    def __init__(self, parameters: dict, sample_size: float = 1.0, view_importance: bool = False):
        self.parameters = parameters
        self.view_importance = view_importance

        train_set, test_set = load_data(sample_size, random_state=42)

        self._train_set_base, self._val_set_base, self._test_set_base = preprocess_data(train_set, test_set)

        self.feature_cols = select_feature_columns(self._train_set_base)

        self.reset_data()

    def reset_data(self) -> None:
        """Restore the pipeline datasets to their clean, prediction-free state."""
        self.train_set = clear_predictions(self._train_set_base)
        self.val_set = clear_predictions(self._val_set_base)
        self.test_set = clear_predictions(self._test_set_base)

    def run(self, approach: Literal['baseline', 'content_knowledge', 'lambdamart', 'ceiling', 'xgboost', 'catboost', 'ensemble']) -> tuple:
        self.reset_data()

        APPROACH_MAP = {
            'baseline': self._run_baseline,
            'content_knowledge': self._run_content_knowledge,   
            'lambdamart': self._run_lambdamart,
            'ceiling': self._run_ceiling,
            'xgboost': self._run_xgboost,
            'catboost': self._run_catboost,
            'ensemble': self._run_ensemble
        }

        clean_train = self.train_set.drop(columns=['date_time', 'checkin_date', 'checkout_date'])
        clean_val = self.val_set.drop(columns=['date_time', 'checkin_date', 'checkout_date'])
        clean_test = self.test_set.drop(columns=['date_time', 'checkin_date', 'checkout_date'])

        return APPROACH_MAP[approach](clean_train, clean_val, clean_test)

    def _run_predictions(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame, predict_func, test_predict_func) -> tuple:
        train_predictions = predict_func(train_set)
        train_acc = compute_accuracy(train_predictions)

        val_predictions = predict_func(val_set)
        val_acc = compute_accuracy(val_predictions)

        test_predictions = test_predict_func(test_set)

        return (
            train_predictions, val_predictions, test_predictions,
            train_acc, val_acc, None
        )

    def _run_ceiling(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):
        return self._run_predictions(train_set, val_set, test_set, predict_func=ceiling, test_predict_func=clear_predictions)

    def _run_baseline(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):
        return self._run_predictions(train_set, val_set, test_set, predict_func=random, test_predict_func=random)

    def _run_lambdamart(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):
        params = self.parameters['lambdamart']
        ranker = LambdaMARTRanker(
            params = params
        )
        ranker.train(train_set, val_set, feature_list=self.feature_cols)
        
        # See feature importance of training evaluation (not validation!)
        if self.view_importance:
            print('Most important features')
            feature_importance_df = ranker.get_feature_importance(importance_type='gain', ascending=False)
            print(feature_importance_df.head(20))
            print('Least important features')
            feature_importance_df = ranker.get_feature_importance(importance_type='gain', ascending=True)
            print(feature_importance_df.head(20))

        return self._run_predictions(
            train_set,
            val_set,
            test_set,
            predict_func=ranker.predict,
            test_predict_func=ranker.predict
        )

    def _run_xgboost(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):        
        params = self.parameters['xgboost']

        ranker = XGBoostRanker(
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            gamma=params['gamma'],
            reg_lambda=params['reg_lambda'],
            reg_alpha=params['reg_alpha']
        )

        ranker.train(train_set, val_set, feature_list=self.feature_cols)
        return self._run_predictions(
            train_set,
            val_set,
            test_set,
            predict_func=ranker.predict,
            test_predict_func=ranker.predict
        )

    def _run_content_knowledge(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):
        recommender = ContentKnowledgeRecommender()
        recommender.train(train_set, val_set)

        return self._run_predictions(
            train_set,
            val_set,
            test_set,
            predict_func=recommender.predict,
            test_predict_func=recommender.predict,
        )
    
    def _run_catboost(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):
        params = self.parameters['catboost']

        ranker = CatBoostRanker(
            iterations=params['iterations'],
            learning_rate=params['learning_rate'],
            depth=params['depth'],
            l2_leaf_reg=params['l2_leaf_reg'],
        )

        ranker.train(train_set, val_set, feature_list=self.feature_cols)
        return self._run_predictions(
            train_set,
            val_set,
            test_set,
            predict_func=ranker.predict,
            test_predict_func=ranker.predict
        )

    def _run_ensemble(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):
        print("\n=== Initializing Elite Ensemble (XGB + LGBM + CatBoost) ===")
        
        # 1. Define your Ensemble Weights
        # These should sum to 1.0. You will tune these later based on validation scores!
        weights = {
            'xgb': 0.6,
            'lgbm': 0.4
        }

        # 2. Initialize All Three Models (Assuming CatBoostRanker is added to predict.py)
        xgb_params = self.parameters.get('xgboost', {})
        lgbm_params = self.parameters.get('lambdamart', {})

        xgb_ranker = XGBoostRanker(**xgb_params)
        lgbm_ranker = LambdaMARTRanker(params = lgbm_params)

        # 3. Train independently
        print(f"[1/3] Training XGBoost (Weight: {weights['xgb']})...")
        xgb_ranker.train(train_set, val_set, feature_list=self.feature_cols)
        
        print(f"[2/3] Training LightGBM (Weight: {weights['lgbm']})...")
        lgbm_ranker.train(train_set, val_set, feature_list=self.feature_cols)

        # 4. The Weighted RRF Prediction Function
        def ensemble_predict(df: pd.DataFrame):
            df_out = df.copy()
            
            # Extract predictions and rename columns safely
            xgb_ranks = xgb_ranker.predict(df)[['srch_id', 'prop_id', 'position']].rename(columns={'position': 'xgb_pos'})
            lgbm_ranks = lgbm_ranker.predict(df)[['srch_id', 'prop_id', 'position']].rename(columns={'position': 'lgbm_pos'})
            
            # Merge all three onto the master dataframe
            df_out = df_out.merge(xgb_ranks, on=['srch_id', 'prop_id'], how='left')
            df_out = df_out.merge(lgbm_ranks, on=['srch_id', 'prop_id'], how='left')
            
            # Calculate the Weighted Reciprocal Rank Fusion Score (k=60 is standard)
            k = 60
            df_out['rrf_score'] = (
                (weights['xgb'] * (1.0 / (k + df_out['xgb_pos']))) +
                (weights['lgbm'] * (1.0 / (k + df_out['lgbm_pos'])))
            )
            
            # Re-rank based on the final weighted score (Higher score = Better rank)
            df_out['position'] = df_out.groupby('srch_id')['rrf_score'].rank(
                method='first', 
                ascending=False
            ).astype('int32')
            
            # Sort and clean up
            df_out = df_out.sort_values(by=['srch_id', 'position']).reset_index(drop=True)
            df_out = df_out.drop(columns=['xgb_pos', 'lgbm_pos', 'rrf_score'])
            
            return df_out

        print("=== Ensemble Training Complete. Running Predictions... ===")
        return self._run_predictions(
            train_set, val_set, test_set,
            predict_func=ensemble_predict,
            test_predict_func=ensemble_predict
        )
