from pathlib import Path
import pandas as pd
import kagglehub

from consts import SRC_DIR

# Create a local cache directory to store the ultra-fast Parquet files
CACHE_DIR = SRC_DIR / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _validate_query_sample_proportion(query_sample_proportion: float | None) -> float | None:
    if query_sample_proportion is None:
        return None

    if query_sample_proportion <= 0 or query_sample_proportion > 1:
        raise ValueError(f"Expected query_sample_proportion to be between 0 and 1, got {query_sample_proportion}")

    return query_sample_proportion


def _full_cache_path(filename: str) -> Path:
    return CACHE_DIR / f"{filename}__full.parquet"


def _sample_cache_path(
    filename: str,
    query_sample_proportion: float,
    random_state: int | None,
) -> Path:
    proportion_token = f"{query_sample_proportion:.6f}".replace('.', 'p')
    random_state_token = "none" if random_state is None else str(random_state)
    return CACHE_DIR / f"{filename}__q{proportion_token}__rs{random_state_token}.parquet"


def _extract_cached_proportion(path: Path) -> float | None:
    parts = path.stem.split("__")
    proportion_part = next((part for part in parts if part.startswith("q")), None)

    if proportion_part is None:
        return None

    try:
        return float(proportion_part[1:].replace("p", ".", 1))
    except ValueError:
        return None


def _find_compatible_cached_sample(
    filename: str,
    requested_proportion: float,
    random_state: int | None,
) -> tuple[Path, float] | None:
    random_state_token = "none" if random_state is None else str(random_state)
    candidates: list[tuple[Path, float]] = []

    for path in CACHE_DIR.glob(f"{filename}__q*__rs{random_state_token}.parquet"):
        cached_proportion = _extract_cached_proportion(path)
        if cached_proportion is None:
            continue
        if cached_proportion >= requested_proportion:
            candidates.append((path, cached_proportion))

    full_path = _full_cache_path(filename)
    if full_path.exists():
        candidates.append((full_path, 1.0))

    if not candidates:
        return None

    return min(candidates, key=lambda candidate: candidate[1])

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

    if query_sample_proportion is None:
        return df

    unique_queries = df['srch_id'].drop_duplicates()
    sampled_queries = unique_queries.sample(frac=query_sample_proportion, random_state=random_state)
    return df[df['srch_id'].isin(sampled_queries)]

def _get_or_create_parquet(csv_path: Path, filename: str) -> pd.DataFrame:
    """
    Checks for a local Parquet version of the dataset. 
    If missing, reads the CSV, parses dates FAST, and saves the Parquet for next time.
    """
    parquet_path = _full_cache_path(filename)
    
    if parquet_path.exists():
        return pd.read_parquet(parquet_path, engine='pyarrow')
    
    df = pd.read_csv(csv_path, engine='pyarrow')
    df['date_time'] = pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S')
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    return df


def _get_or_create_sampled_parquet(
    csv_path: Path,
    filename: str,
    query_sample_proportion: float | None,
    random_state: int | None,
) -> pd.DataFrame:
    query_sample_proportion = _validate_query_sample_proportion(query_sample_proportion)

    if query_sample_proportion is None:
        return _get_or_create_parquet(csv_path, filename)

    target_cache_path = _sample_cache_path(filename, query_sample_proportion, random_state)
    if target_cache_path.exists():
        return pd.read_parquet(target_cache_path, engine='pyarrow')

    compatible_cache = _find_compatible_cached_sample(filename, query_sample_proportion, random_state)

    if compatible_cache is None:
        base_df = _get_or_create_parquet(csv_path, filename)
        sampled_df = _sample_queries(base_df, query_sample_proportion, random_state)
    else:
        compatible_path, compatible_proportion = compatible_cache
        compatible_df = pd.read_parquet(compatible_path, engine='pyarrow')
        relative_proportion = query_sample_proportion / compatible_proportion
        sampled_df = _sample_queries(compatible_df, relative_proportion, random_state)

    sampled_df.to_parquet(target_cache_path, engine='pyarrow', index=False)
    return sampled_df

def load_submission_sample(path: Path | str | None = None) -> pd.DataFrame:
    csv_path = _competition_data_path(path) / 'submission_sample.csv'
    return _get_or_create_parquet(csv_path, 'submission_sample')

def load_training_set(
    path: Path | str | None = None,
    query_sample_proportion: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    csv_path = _competition_data_path(path) / 'training_set_VU_DM.csv'
    return _get_or_create_sampled_parquet(
        csv_path,
        'training_set_VU_DM',
        query_sample_proportion,
        random_state,
    )

def load_test_set(
    path: Path | str | None = None,
    query_sample_proportion: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    csv_path = _competition_data_path(path) / 'test_set_VU_DM.csv'
    return _get_or_create_sampled_parquet(
        csv_path,
        'test_set_VU_DM',
        query_sample_proportion,
        random_state,
    )

def load_data(
    query_sample_proportion: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_training_set(None, query_sample_proportion, random_state),
        load_test_set(None, query_sample_proportion, random_state),
    )