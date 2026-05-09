import numpy as np
import pandas as pd
from numba import njit

from sklearn.metrics import ndcg_score

def create_relevance_target(df):
    """
    Maps clicks and bookings to the Kaggle competition relevance scale.
    Assumes your training data has 'click_bool' and 'booking_bool' columns.
    """
    conditions = [
        df['booking_bool'] == 1,
        df['click_bool'] == 1
    ]
    choices = [5, 1]
    
    # 5 if booked, 1 if clicked (but not booked), 0 otherwise
    df['relevance'] = np.select(conditions, choices, default=0)
    return df

@njit(cache=True)
def _compute_ndcg_bulk(relevances, group_boundaries, k):
    num_groups = len(group_boundaries) - 1
    total_ndcg = 0.0
    valid_groups = 0

    # Pre-compute the log2 discounts to avoid calculating them thousands of times
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    for i in range(num_groups):
        start = group_boundaries[i]
        end = group_boundaries[i+1]
        group_length = end - start

        if group_length == 0:
            continue

        limit = min(k, group_length)
        
        # Calculate actual DCG for this group
        actual_dcg = 0.0
        for j in range(limit):
            actual_dcg += relevances[start + j] * discounts[j]

        # Calculate Ideal DCG for this group
        # Slice the group's relevances, sort descending, and sum
        group_rels = relevances[start:end]
        ideal_rels = np.sort(group_rels)[::-1]
        
        ideal_dcg = 0.0
        for j in range(limit):
            ideal_dcg += ideal_rels[j] * discounts[j]

        # Accumulate normalized score
        if ideal_dcg > 0.0:
            total_ndcg += actual_dcg / ideal_dcg
        
        valid_groups += 1

    return total_ndcg / valid_groups if valid_groups > 0 else 0.0


def compute_loss(df):
    """
    Calculates the average NDCG@k across all search queries.
    Optimized to bypass Pandas groupby overhead.
    """
    # 1. Global sort is exponentially faster than thousands of group-level sorts.
    # Sorting by srch_id ensures groups are contiguous blocks.
    # Sorting by position puts them in the ranked order required for DCG.
    df_sorted = df.sort_values(by=['srch_id', 'position'], ascending=[True, True])
    
    # 2. Extract pure NumPy arrays
    relevances = df_sorted['relevance'].to_numpy(dtype=np.float64)
    srch_ids = df_sorted['srch_id'].to_numpy()

    # 3. Find the starting index boundaries of each unique group
    # np.unique with return_index=True is highly optimized in C
    _, start_indices = np.unique(srch_ids, return_index=True)
    
    # Append the total array length to cap the final group's boundary
    boundaries = np.append(start_indices, len(relevances))
    
    # 4. Execute the fully compiled Numba loop
    return _compute_ndcg_bulk(relevances, boundaries, 5)