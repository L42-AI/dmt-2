from typing import Literal

from data import load_data
from data import preprocess_data
from data import train_val_split
from data import build_relevance_scores

from predict import clear_predictions, random, ceiling, LambdaMARTRanker

from evaluate import compute_loss

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

    def _run_baseline(self, train_set, val_set, test_set):

        train_predictions = random(train_set)
        train_loss = compute_loss(train_predictions)

        val_predictions = random(val_set)
        val_loss = compute_loss(val_predictions)

        test_predictions = random(test_set)
        test_loss = None
        
        return (
            train_predictions, val_predictions, test_predictions,
            train_loss, val_loss, test_loss
            )

    def _run_advanced(self, train_set, val_set, test_set):
        """Run LambdaMART learning-to-rank approach."""
        # Initialize and train the ranker
        ranker = LambdaMARTRanker(num_leaves=31, learning_rate=0.1, n_estimators=100)
        ranker.train(train_set, val_set)

        # Generate predictions
        train_predictions = ranker.predict(train_set)
        train_loss = compute_loss(train_predictions)

        val_predictions = ranker.predict(val_set)
        val_loss = compute_loss(val_predictions)

        test_predictions = ranker.predict(test_set)

        return (
            train_predictions, val_predictions, test_predictions,
            train_loss, val_loss, None
        )

    def _run_ceiling(self, train_set, val_set, test_set):
        """Run the theoretical ceiling predictor on labeled sets."""
        train_predictions = ceiling(train_set)
        train_loss = compute_loss(train_predictions)

        val_predictions = ceiling(val_set)
        val_loss = compute_loss(val_predictions)

        # The test set is unlabeled, so a true ceiling ranking is not defined there.
        test_predictions = clear_predictions(test_set)

        return (
            train_predictions, val_predictions, test_predictions,
            train_loss, val_loss, None
        )
