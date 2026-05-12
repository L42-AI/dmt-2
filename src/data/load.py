from pathlib import Path

import pandas as pd
import kagglehub

def _competition_data_path(path: Path | str | None) -> Path:
    if path is None:
        path = kagglehub.competition_download('dmt-2026-2nd-assignment')
    return Path(path)

def _sample_queries(
    df: pd.DataFrame,
    query_sample_proportion: float | None,
    random_state: int | None = None,
) -> pd.DataFrame:
    
    if query_sample_proportion is None:
        return df

    if 0 > query_sample_proportion or query_sample_proportion > 1:
        raise ValueError(f"Expected query_sample_proportion to be between 0 and 1, got {query_sample_proportion}")

    unique_queries = df['srch_id'].drop_duplicates()
    sampled_queries = unique_queries.sample(frac=query_sample_proportion, random_state=random_state)
    return df[df['srch_id'].isin(sampled_queries)]


def load_submission_sample(path: Path | str | None = None) -> pd.DataFrame:
    """Return the submission sample dataframe from the competition dataset."""
    if path is None:
        path = _competition_data_path(path)
    else:
        path = Path(path)
    return pd.read_csv(path / 'submission_sample.csv', low_memory=False)

def load_training_set(
    path: Path | str | None = None,
    query_sample_proportion: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return the training set dataframe from the competition dataset.

    Set ``query_sample_proportion`` to keep only a random proportion of queries.
    """
    path = _competition_data_path(path)

    df = pd.read_csv(path / 'training_set_VU_DM.csv', low_memory=False, parse_dates=['date_time'])
    return _sample_queries(df, query_sample_proportion, random_state)

def load_test_set(
    path: Path | str | None = None,
    query_sample_proportion: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return the test set dataframe from the competition dataset.

    Set ``query_sample_proportion`` to keep only a random proportion of queries.
    """
    path = _competition_data_path(path)

    df = pd.read_csv(path / 'test_set_VU_DM.csv', low_memory=False, parse_dates=['date_time'])
    return _sample_queries(df, query_sample_proportion, random_state)

def load_data(
    query_sample_proportion: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return training_set and test_set dataframes from the competition dataset.

    Set ``query_sample_proportion`` to keep only a random proportion of queries in both sets.
    """
    path = _competition_data_path(None)
    return (
        load_training_set(path, query_sample_proportion, random_state),
        load_test_set(path, query_sample_proportion, random_state),
    )