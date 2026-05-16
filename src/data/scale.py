import pandas as pd
import numpy as np

def scale_bounded(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manual scaling of features with known ranges.
    """

    # Scale visitor_hist_starrating (1.0 to 5.0)
    df['visitor_hist_starrating'] = (df['visitor_hist_starrating'] - 1.0) / (5.0 - 1.0)

    return df

def clip_persona_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Caps parent and children counts and converts them to discrete integer categories."""
    df['srch_adults_count'] = df['srch_adults_count'].clip(upper=5).astype('uint8')
    df['srch_children_count'] = df['srch_children_count'].clip(upper=4).astype('uint8')
    
    return df

def scale_scores(df: pd.DataFrame, column: str, type: str = 'minmax') -> pd.Series:
    """ Get all scores between 1 and 5 and convert them to 0-1, ignore missing values."""
    series = df[column]
    if type == 'minmax': 
        scaled = (series- series.min()) / (series.max() - series.min())
    if type == 'logminmax':
        log_series = np.log1p(series)
        scaled =(log_series- log_series.min()) / (log_series.max() - log_series.min())
    if type == 'sigmoid':
        z = (series - series.mean()) / series.std()
        scaled = 1 / 1 + np.exp(-z)
        
    return series.where(series.isna(), scaled)
