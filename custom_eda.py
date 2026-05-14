import pandas as pd
import numpy as np
from src.data.load import load_training_set, load_test_set

def run_custom_eda():
    print("Loading data...")
    train = load_training_set(query_sample_proportion=0.05, random_state=42)
    
    train['relevance'] = np.where(train['booking_bool'] == 1, 5, np.where(train['click_bool'] == 1, 1, 0))
    
    print("\n--- 1. Target Variable Distribution ---")
    print(train['relevance'].value_counts(normalize=True) * 100)
    
    print("\n--- 2. Missing Values in Train (%) ---")
    missing = (train.isnull().mean() * 100).sort_values(ascending=False)
    print(missing[missing > 0])
    
    print("\n--- 3. Position Bias (Top 10 Positions) ---")
    pos_stats = train.groupby('position')[['click_bool', 'booking_bool']].mean() * 100
    print(pos_stats.head(10))
    
    print("\n--- 4. Spearman Correlations with Relevance ---")
    exclude = ['position', 'click_bool', 'booking_bool', 'gross_bookings_usd']
    numeric_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c not in exclude]
    corrs = train[numeric_cols].corr(method='spearman')['relevance'].sort_values()
    print("Highest Positive Correlations:")
    print(corrs.dropna().tail(10))
    print("\nHighest Negative Correlations:")
    print(corrs.dropna().head(10))
    
    print("\n--- 6. Random Bool Analysis ---")
    print(train.groupby('random_bool')[['click_bool', 'booking_bool']].mean() * 100)
    
    print("\nPosition bias when random_bool == 1:")
    print(train[train['random_bool'] == 1].groupby('position')[['click_bool', 'booking_bool']].mean().head(10) * 100)
    
    print("\nPosition bias when random_bool == 0:")
    print(train[train['random_bool'] == 0].groupby('position')[['click_bool', 'booking_bool']].mean().head(10) * 100)
    
    print("\nRelevance correlations within random_bool == 1:")
    corrs_rand1 = train[train['random_bool'] == 1][numeric_cols].corr(method='spearman')['relevance'].sort_values()
    print(corrs_rand1.dropna().tail(5))
    
    print("\nRelevance correlations within random_bool == 0:")
    corrs_rand0 = train[train['random_bool'] == 0][numeric_cols].corr(method='spearman')['relevance'].sort_values()
    print(corrs_rand0.dropna().tail(5))

if __name__ == "__main__":
    run_custom_eda()
