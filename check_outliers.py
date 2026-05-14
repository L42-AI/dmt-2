import pandas as pd
import numpy as np
from src.data.load import load_training_set

def check_outliers():
    print("Loading full training set...")
    # Loading a larger sample to ensure we catch the extreme outliers
    df = load_training_set(query_sample_proportion=1.0)
    
    print("\n--- Summary of Key Numerical Features ---")
    cols_to_check = [
        'price_usd', 'srch_length_of_stay', 'srch_booking_window', 
        'srch_adults_count', 'srch_children_count', 'srch_room_count',
        'prop_starrating', 'prop_review_score', 'prop_location_score1', 
        'prop_location_score2', 'orig_destination_distance', 'visitor_hist_adr_usd'
    ]
    
    # Filter columns that exist
    cols = [c for c in cols_to_check if c in df.columns]
    
    # Calculate percentiles and min/max
    stats = df[cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99, 0.999, 0.9999]).T
    
    # Format output for readability
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(stats[['min', 'max', '50%', '99%', '99.9%', '99.99%']])

    print("\n--- Specific Extreme Value Checks ---")
    print(f"Max price_usd: {df['price_usd'].max():.2f}")
    print(f"Number of rows with price_usd > 10,000: {(df['price_usd'] > 10000).sum()}")
    print(f"Number of rows with price_usd > 100,000: {(df['price_usd'] > 100000).sum()}")
    print(f"Number of rows with price_usd == 0: {(df['price_usd'] == 0).sum()}")
    
if __name__ == "__main__":
    check_outliers()
