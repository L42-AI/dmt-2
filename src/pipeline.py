from typing import Literal

import pandas as pd

from data import load_data
from data import preprocess_data
from data import train_val_split
from data import build_relevance_scores
from data.feature_selection import select_feature_columns

from predict import (
    clear_predictions,
    random,
    ceiling,
    LambdaMARTRanker,
    XGBoostRanker,
    ContentKnowledgeRecommender
)

from evaluate import compute_accuracy

class Pipeline:

    TRAIN_RATIO = 0.8

    

    def __init__(self, parameters: dict, sample_size: float = 1):
        self.parameters = parameters

        train_set, test_set = load_data(sample_size, random_state=42)
        train_set = build_relevance_scores(train_set)
        train_set, val_set = train_val_split(train_set, self.TRAIN_RATIO)

        self._train_set_base = preprocess_data(train_set)
        self._val_set_base = preprocess_data(val_set)
        self._test_set_base = preprocess_data(test_set)

        self.feature_cols = select_feature_columns(self._train_set_base)

        self.reset_data()

    def reset_data(self) -> None:
        """Restore the pipeline datasets to their clean, prediction-free state."""
        self.train_set = clear_predictions(self._train_set_base)
        self.val_set = clear_predictions(self._val_set_base)
        self.test_set = clear_predictions(self._test_set_base)

    def run(self, approach: Literal['baseline', 'content_knowledge', 'lambdamart', 'ceiling', 'xgboost']) -> tuple:
        self.reset_data()

        APPROACH_MAP = {
            'baseline': self._run_baseline,
            'content_knowledge': self._run_content_knowledge,   
            'lambdamart': self._run_lambdamart,
            'ceiling': self._run_ceiling,
            'xgboost': self._run_xgboost,
        }

        return APPROACH_MAP[approach](self.train_set.drop(columns=['date_time', 'checkin_date', 'checkout_date']), self.val_set.drop(columns=['date_time', 'checkin_date', 'checkout_date']), self.test_set.drop(columns=['date_time', 'checkin_date', 'checkout_date']))

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
        ranker = LambdaMARTRanker(num_leaves=params['num_leaves'], learning_rate=params['learning_rate'], n_estimators=params['n_estimators'])
        ranker.train(train_set, val_set)

        return self._run_predictions(train_set, val_set, test_set, predict_func=ranker.predict, test_predict_func=ranker.predict)

    def _run_xgboost(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):        
        ranker = XGBoostRanker()
        ranker.train(train_set, val_set)
        return self._run_predictions(train_set, val_set, test_set, predict_func=ranker.predict, test_predict_func=ranker.predict)

    def _run_content_knowledge(self, train_set, val_set, test_set):
        recommender = ContentKnowledgeRecommender()
        recommender.train(train_set, val_set)

        return self._run_predictions(
        train_set,
        val_set,
        test_set,
        predict_func=recommender.predict,
        test_predict_func=recommender.predict,
    )