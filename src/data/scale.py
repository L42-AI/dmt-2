import pandas as pd

def scale_bounded(df: pd.DataFrame) -> pd.DataFrame:
    """
    Manual scaling of features with known ranges.
    """

    # Scale visitor_hist_starrating (1.0 to 5.0)
    df['visitor_hist_starrating'] = (df['visitor_hist_starrating'] - 1.0) / (5.0 - 1.0)

    return df