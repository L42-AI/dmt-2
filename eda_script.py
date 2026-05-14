import sys
import os
import pandas as pd
import numpy as np
from src.data.load import load_training_set

def run_eda():
    # Load a small sample to speed up EDA
    print("Loading data sample...")
    df = load_training_set(query_sample_proportion=0.01, random_state=42)
    print(f"Loaded {len(df)} rows across {df['srch_id'].nunique()} queries.")
    
    print("\nMissing values percentage:")
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(missing[missing > 0])
    
    # Calculate a simple relevance proxy
    df['relevance'] = np.where(df['booking_bool'] == 1, 5, np.where(df['click_bool'] == 1, 1, 0))
    
    print("\nCorrelations with relevance:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['relevance'].sort_values()
    print("Bottom 10:")
    print(correlations.head(10))
    print("\nTop 10:")
    print(correlations.tail(11)[:-1]) # exclude relevance itself
    
    print("\nClick/Book rates by position:")
    pos_stats = df.groupby('position')[['click_bool', 'booking_bool']].mean().head(10)
    print(pos_stats)
    
if __name__ == "__main__":
    run_eda()
