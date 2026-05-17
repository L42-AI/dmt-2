import pandas as pd
import numpy as np
from scipy.sparse.csgraph import shortest_path

class GraphDistanceImputer:
    """
    Learns geographic medians and builds a Shortest-Path Triangulation map 
    strictly from the training set to prevent data leakage.
    """
    _N_DISTANCE_BUCKETS = 50

    def __init__(self):
        self.dest_medians = None
        self.prop_medians = None
        self.global_median = None
        self.world_map = None
        self.max_known_id = 0
        self.distance_bin_edges_ = None  # fitted on train, reused for val/test

    def fit(self, train_df: pd.DataFrame) -> "GraphDistanceImputer":
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
        
        v_ids = country_edges['visitor_location_country_id'].astype(int).values
        p_ids = country_edges['prop_country_id'].astype(int).values
        dists = country_edges['orig_destination_distance'].values
        raw_map[v_ids, p_ids] = dists
        raw_map[p_ids, v_ids] = dists

        # Compute shortest paths globally across the training map
        self.world_map = shortest_path(csgraph=raw_map, directed=False, method='FW')

        # --- Fit bucket bin edges on train distribution ---
        _, self.distance_bin_edges_ = pd.qcut(
            train_df['orig_destination_distance'].dropna(),
            q=self._N_DISTANCE_BUCKETS,
            retbins=True,
            duplicates='drop',
        )
        self.distance_bin_edges_[0]  = -np.inf
        self.distance_bin_edges_[-1] =  np.inf

        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned medians and the graph map to fill NaNs."""
        df = df.copy()

        df['imputed_distance'] = df['orig_destination_distance']

        # 1. Apply Level A Medians
        df = df.join(self.dest_medians.rename('median_dest'), on=['visitor_location_country_id', 'srch_destination_id'])
        df['imputed_distance'] = df['imputed_distance'].fillna(df.pop('median_dest'))

        # 2. Apply Level B Medians
        df = df.join(self.prop_medians.rename('median_prop'), on=['visitor_location_country_id', 'prop_country_id'])
        df['imputed_distance'] = df['imputed_distance'].fillna(df.pop('median_prop'))

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
        df['imputed_distance'] = df['imputed_distance'].replace([np.inf, -np.inf], np.nan).fillna(self.global_median)

        # 5. In-Search Relative Z-Score
        grp = df.groupby('srch_id')['imputed_distance']
        df['distance_z_score'] = (
            (df['imputed_distance'] - grp.transform('mean'))
            / grp.transform('std').replace(0, 1)
        )

        # 6. Bucketization with train-fitted edges (consistent across splits)
        df['distance_bucket'] = pd.cut(
            df['imputed_distance'],
            bins=self.distance_bin_edges_,
            labels=False,
        )

        return df