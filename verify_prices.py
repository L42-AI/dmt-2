import pandas as pd
import numpy as np
from src.data.load import load_training_set

def verify_price_behavior():
    print("Loading data...")
    df = load_training_set(query_sample_proportion=0.2, random_state=42) # 20% is enough to find overlaps
    
    print("\n--- 1. Country UI Conventions (Per Night vs Per Stay) ---")
    # Get countries with at least 1000 searches
    country_counts = df['visitor_location_country_id'].value_counts()
    top_countries = country_counts[country_counts > 1000].index
    
    correlations = []
    for c in top_countries:
        country_data = df[df['visitor_location_country_id'] == c]
        # Spearman correlation between price and stay length
        corr = country_data['price_usd'].corr(country_data['srch_length_of_stay'], method='spearman')
        correlations.append({'country': c, 'corr': corr, 'searches': len(country_data)})
    
    corr_df = pd.DataFrame(correlations).sort_values('corr')
    print("Countries with LOWEST correlation (Likely 'Per Night' convention):")
    print(corr_df.head(5).to_string(index=False))
    print("\nCountries with HIGHEST correlation (Likely 'Per Stay' convention):")
    print(corr_df.tail(5).to_string(index=False))

    print("\n--- 2. Same-Property Check ---")
    # Find a popular property
    popular_prop = df['prop_id'].value_counts().index[0]
    prop_data = df[df['prop_id'] == popular_prop][['srch_length_of_stay', 'price_usd', 'visitor_location_country_id']].head(10)
    print(f"Prices for most popular property (ID {popular_prop}) across different searches:")
    print(prop_data.sort_values('srch_length_of_stay').to_string(index=False))

    print("\n--- 3. Verifying High Price Glitches ---")
    high_prices = df[df['price_usd'] > 100000].copy()
    if not high_prices.empty:
        high_prices['price_per_night_per_room'] = high_prices['price_usd'] / (high_prices['srch_length_of_stay'] * high_prices['srch_room_count'])
        print(f"Found {len(high_prices)} rows > $100,000.")
        print("Top 5 highest implied per-night, per-room rates:")
        print(high_prices[['price_usd', 'srch_length_of_stay', 'srch_room_count', 'price_per_night_per_room']].sort_values('price_per_night_per_room', ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    verify_price_behavior()
