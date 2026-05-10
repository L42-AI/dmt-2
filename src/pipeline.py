from typing import Literal

from data import load_data
from data import train_val_split
from data import create_relevance_target

from predict import random

from evaluate import compute_loss

class Pipeline:

    TRAIN_RATIO = 0.8

    def __init__(self):
        self.train_set, self.test_set = load_data(0.2)
        self.train_set = create_relevance_target(self.train_set)

    def run(self, approach: Literal['baseline']) -> tuple:

        self.train_set, self.val_set = train_val_split(self.train_set, self.TRAIN_RATIO)

        match approach:
            case 'baseline':
                return self._run_baseline(self.train_set, self.val_set, self.test_set)
            case 'advanced':
                return self._run_advanced(self.train_set, self.val_set, self.test_set)
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
        pass
