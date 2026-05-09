import math

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
        penalized_top_k = [2 ** top_k[i-1] / math.log2(i + 1) for i in ranks]
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

def evaluate(model, data):
    # Dummy implementation for evaluation
    return {"rmse": 0.0}
