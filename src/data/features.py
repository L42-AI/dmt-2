import pandas as pd

def compute_comp_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the competition rate for each search.
    """
    for num in range(1, 9):
        percent_diff = pd.to_numeric(df[f'comp{num}_rate_percent_diff'], errors='coerce')
        price_usd = pd.to_numeric(df['price_usd'], errors='coerce')
        df[f'comp{num}_rate'] = percent_diff * price_usd

    return df