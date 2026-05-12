import numpy as np
import pandas as pd

__all__ = ["ceiling"]


def ceiling(test_set: pd.DataFrame) -> pd.DataFrame:
    df = test_set

    # Stable tie-breakers keep the output deterministic when labels are equal.
    df["ceiling_tie_breaker"] = np.arange(len(df))
    df.sort_values(
        by=["srch_id", "relevance", "ceiling_tie_breaker"],
        ascending=[True, False, True],
        inplace=True,
    )

    df["position"] = df.groupby("srch_id").cumcount() + 1
    df.drop(columns=["ceiling_tie_breaker"], inplace=True)

    return df