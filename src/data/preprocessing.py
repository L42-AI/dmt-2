import numpy as np
import pandas as pd

from .split import train_val_split
from .impute import Imputer
from .engineer import FeatureEngineer

from .features.ids import resample_ids
from .features.distance import GraphDistanceImputer
from .features.features import convert_target_to_relevance_scores, PropertyStatsTransformer


def preprocess_data(train_set: pd.DataFrame, test_set: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline. Stage order is strict:

        GraphDistanceImputer  — fills missing distances using train geography
        Imputer               — cleans and imputes all raw columns
        PropertyStatsTransformer — per-property stats
        FeatureEngineer       — stateless feature construction using clean, scaled inputs
    """
    # Remap IDs to contiguous integers across the combined ID space.
    # Required for GraphDistanceImputer's array-indexed adjacency matrix.
    complete_set = pd.concat([train_set, test_set], ignore_index=True)
    complete_set = resample_ids(complete_set)
    train_set = complete_set.iloc[:len(train_set)].copy()
    test_set  = complete_set.iloc[len(train_set):].copy()
    del complete_set

    train_set = convert_target_to_relevance_scores(train_set)
    train_set, val_set = train_val_split(train_set, 0.8)

    # Stage 1
    graph_imputer = GraphDistanceImputer()
    graph_imputer.fit(train_set)
    train_set = graph_imputer.transform(train_set)
    val_set   = graph_imputer.transform(val_set)
    test_set  = graph_imputer.transform(test_set)

    # Stage 2
    imputer = Imputer()
    train_set = imputer.fit_transform(train_set)
    val_set   = imputer.transform(val_set)
    test_set  = imputer.transform(test_set)

    # Stage 3
    prop_stats = PropertyStatsTransformer()
    train_set = prop_stats.fit_transform(train_set)
    val_set   = prop_stats.transform(val_set)
    test_set  = prop_stats.transform(test_set)

    # Stage 4
    engineer = FeatureEngineer()
    train_set = engineer.fit_transform(train_set)
    val_set   = engineer.transform(val_set)
    test_set  = engineer.transform(test_set)

    return train_set, val_set, test_set


def select_randomized_instances(df: pd.DataFrame) -> pd.DataFrame:
    return df.query('random_bool == 1')
