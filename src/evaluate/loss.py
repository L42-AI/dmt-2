import math
import numpy as np
import pandas as pd
from numba import njit

NDCG_K = 5

__all__ = [
    "compute_accuracy",
    "dcg",
    "ndcg_5",
]

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


def compute_accuracy(df: pd.DataFrame) -> float:
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

def dcg(relevance_scores: list, top_ranks: int = 5, alt: bool = False) -> float:
    """ Discounted cumulative gain score for a ranked list of results in a search query.

    Args:
        relevance_scores (list):    Relevance scores of ranked properties in a query. 
        top_ranks (int, optional):  The top-k ranks to look for relevance scores. Defaults to 5.
        alt (bool, optional):       Alternative DCG formulation, with stronger empathis on retrieving
                                    relevant documents. Defaults to False.

    Returns:
        float: DCG score for search query results
    """
    top_k = relevance_scores[:top_ranks]
    ranks = range(1, top_ranks + 1) # [1, k]
    if alt:
        penalized_top_k = [(2 ** top_k[i-1] - 1) / math.log2(i + 1) for i in ranks]
    else:
        penalized_top_k = [top_k[i - 1] / math.log2(i + 1) for i in ranks]
    return sum(penalized_top_k)

def ndcg_5(relevance_scores: list, alt: bool = False) -> float:
    """ Normalized discounted cumulative gain for a ranked search query

    Args:
        relevance_scores (list): The relevance scores of the ranked search query
        alt (bool, optional): Alternative DCG formulation. Defaults to False.

    Returns:
        float: NDCG value for search query results.
    """
    dcg_5 = dcg(relevance_scores, 5, alt)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg_5 = dcg(ideal_relevance_scores, 5, alt)
    return dcg_5 / idcg_5