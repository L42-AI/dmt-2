import pandas as pd

def scale_bounded(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manual scaling of features with known ranges.
    """

    # Scale visitor_hist_starrating (1.0 to 5.0)
    df['visitor_hist_starrating'] = (df['visitor_hist_starrating'] - 1.0) / (5.0 - 1.0)

    return df

def clip_persona_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Caps parent and children counts and converts them to discrete integer categories."""
    df = df.copy()
    
    df['srch_adults_count'] = df['srch_adults_count'].clip(upper=5).astype(int)
    df['srch_children_count'] = df['srch_children_count'].clip(upper=4).astype(int)
    
    return df

def scale_scores(df: pd.DataFrame, column: str) -> pd.Series:
    """ Get all scores between 1 and 5 and convert them to 0-1, ignore missing values."""
    series = df[column]
    scaled = (series - 1.0) / (5.0 - 1.0)
    return series.where(series.isna(), scaled)