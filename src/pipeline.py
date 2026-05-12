from typing import Literal

from data import load_data
from data import preprocess_data
from data import train_val_split
from data import build_relevance_scores

from predict import clear_predictions, random, ceiling, LambdaMARTRanker

from evaluate import compute_accuracy

class Pipeline:

    TRAIN_RATIO = 0.8

    def __init__(self):
        train_set, test_set = load_data(0.05)
        train_set = build_relevance_scores(train_set)
        train_set, val_set = train_val_split(train_set, self.TRAIN_RATIO)

        self._train_set_base = preprocess_data(train_set)
        self._val_set_base = preprocess_data(val_set)
        self._test_set_base = preprocess_data(test_set)

        self.reset_data()

    def reset_data(self) -> None:
        """Restore the pipeline datasets to their clean, prediction-free state."""
        self.train_set = clear_predictions(self._train_set_base)
        self.val_set = clear_predictions(self._val_set_base)
        self.test_set = clear_predictions(self._test_set_base)

    def run(self, approach: Literal['baseline', 'advanced', 'ceiling']) -> tuple:
        self.reset_data()

        match approach:
            case 'baseline':
                return self._run_baseline(self.train_set, self.val_set, self.test_set)
            case 'advanced':
                return self._run_advanced(self.train_set, self.val_set, self.test_set)
            case 'ceiling':
                return self._run_ceiling(self.train_set, self.val_set, self.test_set)
            case _:
                raise ValueError(f"Unknown approach: {approach}")

    def _run_predictions(self, train_set, val_set, test_set, predict_func, test_predict_func) -> tuple:
        train_predictions = predict_func(train_set)
        train_acc = compute_accuracy(train_predictions)

        val_predictions = predict_func(val_set)
        val_acc = compute_accuracy(val_predictions)

        test_predictions = test_predict_func(test_set)

        return (
            train_predictions, val_predictions, test_predictions,
            train_acc, val_acc, None
        )

    def _run_ceiling(self, train_set, val_set, test_set):
        return self._run_predictions(train_set, val_set, test_set, predict_func=ceiling, test_predict_func=clear_predictions)

    def _run_baseline(self, train_set, val_set, test_set):
        return self._run_predictions(train_set, val_set, test_set, predict_func=random, test_predict_func=random)

    def _run_advanced(self, train_set, val_set, test_set):
        ranker = LambdaMARTRanker(num_leaves=31, learning_rate=0.1, n_estimators=100)
        ranker.train(train_set, val_set)

        return self._run_predictions(train_set, val_set, test_set, predict_func=ranker.predict, test_predict_func=ranker.predict)