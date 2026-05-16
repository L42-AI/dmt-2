import pandas as pd
import numpy as np
from sklearn import set_config
set_config(transform_output="pandas")
from sklearn.preprocessing import MinMaxScaler

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

class Scaler:
    def __init__(self, exclude):
        self.minmax_scaler = MinMaxScaler()
        self.exclude = set(exclude)  # Set for faster lookups

    def fit_transform(self, train_set: pd.DataFrame, val_set: pd.DataFrame, test_set: pd.DataFrame):
        # Identify columns to scale all at once
        cols_to_scale = [col for col in train_set.columns if col not in self.exclude]
        # If there's nothing to scale, just return the dataframes
        if not cols_to_scale:
            return train_set, val_set, test_set

        # Fit and transform 
        train_set[cols_to_scale] = self.minmax_scaler.fit_transform(train_set[cols_to_scale])
        val_set[cols_to_scale] = self.minmax_scaler.transform(val_set[cols_to_scale])
        test_set[cols_to_scale] = self.minmax_scaler.transform(test_set[cols_to_scale])

        return train_set, val_set, test_set



