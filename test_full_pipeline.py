import numpy as np
from src.data.load import load_training_set
from src.data.preprocessing import preprocess_data

df = load_training_set(query_sample_proportion=0.01, random_state=42)
df = preprocess_data(df)

issues = []
for col in df.select_dtypes(include=[np.number]).columns:
    if np.isinf(df[col]).any():
        issues.append(f"INF in column: {col}")

if issues:
    print("Found issues:")
    for issue in issues:
        print(issue)
else:
    print("No INF values found.")
