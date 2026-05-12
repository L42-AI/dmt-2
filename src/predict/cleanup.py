import pandas as pd

__all__ = ["clear_predictions"]


PREDICTION_COLUMNS = (
    "position",
    "random_score",
    "lambdamart_score",
    "ceiling_tie_breaker",
)


def clear_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns created by prediction helpers and return a clean copy."""
    columns_to_drop = [column for column in PREDICTION_COLUMNS if column in df.columns]
    if not columns_to_drop:
        return df.copy()

    df.drop(columns=columns_to_drop, inplace=True)
    return df