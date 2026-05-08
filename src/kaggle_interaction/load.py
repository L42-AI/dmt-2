from pathlib import Path

import pandas as pd
import kagglehub

def load_submission_sample(path: str | None = None) -> pd.DataFrame:
    """ Returns the submission sample dataframe from the competition dataset. """
    if path is None:
        path = Path(kagglehub.competition_download('dmt-2026-2nd-assignment'))
    return pd.read_csv(path / 'submission_sample.csv')

def load_training_set(path: str | None = None) -> pd.DataFrame:
    """ Returns the training set dataframe from the competition dataset. """
    if path is None:
        path = Path(kagglehub.competition_download('dmt-2026-2nd-assignment'))
    return pd.read_csv(path / 'training_set_VU_DM.csv')

def load_test_set(path: str | None = None) -> pd.DataFrame:
    """ Returns the test set dataframe from the competition dataset. """
    if path is None:
        path = Path(kagglehub.competition_download('dmt-2026-2nd-assignment'))
    return pd.read_csv(path / 'test_set_VU_DM.csv')

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Returns training_set, test_set dataframes from the competition dataset. """
    path = Path(kagglehub.competition_download('dmt-2026-2nd-assignment'))
    return (
        load_training_set(path),
        load_test_set(path)
    )