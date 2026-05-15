from pathlib import Path
import pandas as pd
import kagglehub

from consts import SRC_DIR

CACHE_DIR = SRC_DIR / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _validate_query_sample_proportion(query_sample_proportion: float | None) -> float | None:
    if query_sample_proportion is None:
        return None

    if query_sample_proportion <= 0 or query_sample_proportion > 1:
        raise ValueError(f"Expected query_sample_proportion to be between 0 and 1, got {query_sample_proportion}")

    return query_sample_proportion

def _cache_path(filename: str) -> Path:
    return CACHE_DIR / f"{filename}.parquet"

def _competition_data_path(path: Path | str | None) -> Path:
    if path is None:
        path = kagglehub.competition_download('dmt-2026-2nd-assignment')
    return Path(path)


def _sample_queries(
    df: pd.DataFrame,
    query_sample_proportion: float | None,
    random_state: int | None = None,
) -> pd.DataFrame:
    query_sample_proportion = _validate_query_sample_proportion(query_sample_proportion)

    if query_sample_proportion is None or query_sample_proportion == 1.0:
        return df

    unique_queries = df['srch_id'].drop_duplicates()
    sampled_queries = unique_queries.sample(frac=query_sample_proportion, random_state=random_state)
    return df[df['srch_id'].isin(sampled_queries)]


def _get_or_create_parquet(csv_path: Path, filename: str) -> pd.DataFrame:
    """
    Checks for a local Parquet version of the FULL dataset. 
    If missing, reads the CSV, parses dates FAST, and saves the Parquet for next time.
    """
    parquet_path = _cache_path(filename)
    
    if parquet_path.exists():
        return pd.read_parquet(parquet_path, engine='pyarrow')
    
    df = pd.read_csv(csv_path, engine='pyarrow')
    df['date_time'] = pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S')
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    return df


def load_submission_sample(path: Path | str | None = None) -> pd.DataFrame:
    csv_path = _competition_data_path(path) / 'submission_sample.csv'
    return _get_or_create_parquet(csv_path, 'submission_sample')


def load_training_set(
    path: Path | str | None = None,
    query_sample_proportion: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    csv_path = _competition_data_path(path) / 'training_set_VU_DM.csv'
    
    # 1. Always load the full cached dataset first
    df = _get_or_create_parquet(csv_path, 'training_set_VU_DM')
    
    # 2. Sample on the fly
    return _sample_queries(df, query_sample_proportion, random_state)


def load_test_set(
    path: Path | str | None = None,
    query_sample_proportion: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    csv_path = _competition_data_path(path) / 'test_set_VU_DM.csv'
    
    # 1. Always load the full cached dataset first
    df = _get_or_create_parquet(csv_path, 'test_set_VU_DM')
    
    # 2. Sample on the fly
    return _sample_queries(df, query_sample_proportion, random_state)


def load_data(
    query_sample_proportion: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_training_set(None, query_sample_proportion, random_state),
        load_test_set(None, query_sample_proportion, random_state),
    )