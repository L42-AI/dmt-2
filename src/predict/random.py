import pandas as pd
import numpy as np

def random(test_set: pd.DataFrame) -> dict[int, list[int]]:
    predictions: dict[int, list[int]] = {}

    for srch_id, grouped_df in test_set.groupby('srch_id'):
        prop_ids = grouped_df['prop_id'].astype(int).to_numpy(copy=True)
        np.random.shuffle(prop_ids)
        predictions[srch_id] = prop_ids.tolist()

    return predictions