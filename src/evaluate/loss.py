import numpy as np
import pandas as pd
from numba import njit

NDCG_K = 5

__all__ = ["compute_loss"]

@njit(cache=True)
def _compute_ndcg_bulk(relevances, group_boundaries, k):
    """Compute mean NDCG@k over contiguous query groups."""
    num_groups = len(group_boundaries) - 1
    total_ndcg = 0.0
    valid_groups = 0

    # Pre-compute discount factors once and reuse for each group.
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    for group_idx in range(num_groups):
        start = group_boundaries[group_idx]
        end = group_boundaries[group_idx + 1]
        group_size = end - start

        if group_size == 0:
            continue

        rank_limit = min(k, group_size)

        # DCG for the current ranking (already sorted by position).
        actual_dcg = 0.0
        for rank in range(rank_limit):
            actual_dcg += relevances[start + rank] * discounts[rank]

        # Ideal DCG for this group (same labels, perfect descending order).
        group_relevances = relevances[start:end]
        ideal_relevances = np.sort(group_relevances)[::-1]

        ideal_dcg = 0.0
        for rank in range(rank_limit):
            ideal_dcg += ideal_relevances[rank] * discounts[rank]

        # Keep existing behavior: groups with zero ideal DCG count as zero contribution.
        if ideal_dcg > 0.0:
            total_ndcg += actual_dcg / ideal_dcg

        valid_groups += 1

    return total_ndcg / valid_groups if valid_groups > 0 else 0.0


def compute_loss(df: pd.DataFrame) -> float:
    """Calculate mean NDCG@5 across all search queries in the input dataframe."""
    # Sort globally once so each query is contiguous and already in rank order.
    df_sorted = df.sort_values(by=["srch_id", "position"], ascending=[True, True])

    # Extract contiguous arrays for fast compiled iteration.
    relevances = df_sorted['relevance'].to_numpy(dtype=np.float64)
    srch_ids = df_sorted['srch_id'].to_numpy()

    # Locate start index of each query block.
    _, start_indices = np.unique(srch_ids, return_index=True)

    # Append array length so the final group has a closing boundary.
    boundaries = np.append(start_indices, len(relevances))

    return _compute_ndcg_bulk(relevances, boundaries, NDCG_K)