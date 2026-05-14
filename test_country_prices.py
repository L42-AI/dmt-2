import pandas as pd
from src.data.load import load_training_set

df = load_training_set(query_sample_proportion=0.2, random_state=42)
country_counts = df['visitor_location_country_id'].value_counts()
top_countries = country_counts[country_counts > 1000].index

print("Looking at median prices by length of stay for Country 215 (Highly correlated) vs Country 15 (Low correlation)")
for c in [215, 15]:
    c_df = df[df['visitor_location_country_id'] == c]
    print(f"\nCountry {c}:")
    print(c_df.groupby('srch_length_of_stay')['price_usd'].median().head(7))
