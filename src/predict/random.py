import numpy as np
import pandas as pd

__all__ = ["random"]

def random(test_set: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a random ranking position to each property within a search query.
    Fully vectorized to bypass Python loops.
    """
    df = test_set

    # 1. Generate a random float for every single row instantly in C
    df['random_score'] = np.random.rand(len(df))

    # 2. Sort the dataframe globally by the search ID and this random score.
    # This naturally shuffles the properties within each search group.
    df.sort_values(by=['srch_id', 'random_score'], inplace=True)

    # 3. Assign positions 1 to N using cumulative count within each sorted group
    df['position'] = df.groupby('srch_id').cumcount() + 1

    # 4. Clean up the temporary column
    df.drop(columns=['random_score'], inplace=True)
    
    return df