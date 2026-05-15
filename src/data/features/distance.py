import pandas as pd
import numpy as np
from scipy.sparse.csgraph import shortest_path

class GraphDistanceImputer:
    """
    Learns geographic medians and builds a Shortest-Path Triangulation map 
    strictly from the training set to prevent data leakage.
    """
    def __init__(self):
        self.dest_medians = None
        self.prop_medians = None
        self.global_median = None
        self.world_map = None
        self.max_known_id = 0

    def fit(self, train_df: pd.DataFrame) -> None:
        """Learn all medians and build the geographic network from Train data ONLY."""
        
        # 1. Learn the Hierarchical Medians
        self.dest_medians = train_df.groupby(
            ['visitor_location_country_id', 'srch_destination_id']
        )['orig_destination_distance'].median()

        self.prop_medians = train_df.groupby(
            ['visitor_location_country_id', 'prop_country_id']
        )['orig_destination_distance'].median()

        self.global_median = train_df['orig_destination_distance'].median()

        # 2. Build the Triangulation Graph
        country_edges = train_df.dropna(subset=['orig_destination_distance']).groupby(
            ['visitor_location_country_id', 'prop_country_id']
        )['orig_destination_distance'].median().reset_index()
        
        self.max_known_id = int(max(
            train_df['visitor_location_country_id'].max(), 
            train_df['prop_country_id'].max()
        )) + 1
        
        raw_map = np.full((self.max_known_id, self.max_known_id), np.inf)
        np.fill_diagonal(raw_map, 0)
        
        for _, row in country_edges.iterrows():
            v_id, p_id = int(row['visitor_location_country_id']), int(row['prop_country_id'])
            dist = row['orig_destination_distance']
            raw_map[v_id, p_id] = dist
            raw_map[p_id, v_id] = dist 
            
        # Compute shortest paths globally across the training map
        self.world_map = shortest_path(csgraph=raw_map, directed=False, method='FW')

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned medians and the graph map to fill NaNs."""
        df = df.copy()

        df['imputed_distance'] = df['orig_destination_distance']

        # 1. Apply Level A Medians
        df = df.join(self.dest_medians.rename('median_dest'), on=['visitor_location_country_id', 'srch_destination_id'])
        df['imputed_distance'] = df['imputed_distance'].fillna(df['median_dest'])
        df = df.drop(columns=['median_dest'])

        # 2. Apply Level B Medians
        df = df.join(self.prop_medians.rename('median_prop'), on=['visitor_location_country_id', 'prop_country_id'])
        df['imputed_distance'] = df['imputed_distance'].fillna(df['median_prop'])
        df = df.drop(columns=['median_prop'])

        # 3. Apply the Graph Triangulation
        missing_mask = df['imputed_distance'].isna()
        
        if missing_mask.any():
            # Identify which missing rows actually contain KNOWN country IDs
            # (If the ID is >= max_known_id, or is NaN, it evaluates to False)
            known_v_ids = df['visitor_location_country_id'] < self.max_known_id
            known_p_ids = df['prop_country_id'] < self.max_known_id
            
            # Combine the masks: It must be missing a distance, AND both IDs must be known
            safe_to_triangulate_mask = missing_mask & known_v_ids & known_p_ids
            
            if safe_to_triangulate_mask.any():
                # Safely extract the valid IDs without needing to clip
                v_ids = df.loc[safe_to_triangulate_mask, 'visitor_location_country_id'].astype(int).values
                p_ids = df.loc[safe_to_triangulate_mask, 'prop_country_id'].astype(int).values
                
                # Perform the lookup and apply only to the safe rows
                triangulated_distances = self.world_map[v_ids, p_ids]
                df.loc[safe_to_triangulate_mask, 'imputed_distance'] = triangulated_distances

        # 4. Final Global Fallback
        df['imputed_distance'] = df['imputed_distance'].replace([np.inf, -np.inf], np.nan)

        # 5. In-Search Relative Z-Score
        std = df.groupby('srch_id')['imputed_distance'].transform('std').replace(0, 1)
        df['distance_z_score'] = (df['imputed_distance'] - df.groupby('srch_id')['imputed_distance'].transform('mean')) / std

        df = add_distance_bucketization(df)

        return df
    
def add_distance_bucketization(df: pd.DataFrame, num_buckets: int = 50) -> pd.DataFrame:
    """
    Buckets the continuous result into distinct quantiles.
    """

    # Create quantile-based buckets for the imputed distance, excluding the -1 values
    df['distance_bucket'] = pd.qcut(
        df['imputed_distance'],
        q=num_buckets,
        labels=False,
        duplicates='drop'
    )

    return df