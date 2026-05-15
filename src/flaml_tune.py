import pandas as pd
from flaml import AutoML
import json

# Assuming you have a function to get your preprocessed train/val data
from data import load_data, preprocess_data

def fast_tune_xgboost_with_flaml(sample_size: float = 1, time_budget_seconds: int = 500):
    print(f"=== INITIALIZING FLAML AUTOML (Budget: {time_budget_seconds}s) ===")
    
    # 1. Load and prep data
    train_set, test_set = load_data(sample_size, random_state=42)
    train_df, val_df, _ = preprocess_data(train_set, test_set)
    
    # CRITICAL: For ranking, the dataframe MUST be sorted by the query ID
    train_df = train_df.sort_values('srch_id')
    val_df = val_df.sort_values('srch_id')
    
    # 2. Extract Features, Labels, and Groups
    drop_cols = ['relevance', 'position', 'click_bool', 'booking_bool', 'srch_id', 'prop_id', 'date_time', 'checkin_date', 'checkout_date']
    
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['relevance']
    # The 'groups' array tells FLAML how many rows belong to each search
    groups_train = train_df.groupby('srch_id').size().values
    
    X_val = val_df.drop(columns=drop_cols, errors='ignore')
    y_val = val_df['relevance']
    groups_val = val_df.groupby('srch_id').size().values

    # 3. Configure FLAML
    automl = AutoML()
    
    flaml_settings = {
        "time_budget": time_budget_seconds,  # FLAML stops exactly when the clock runs out
        "metric": "ndcg@5",                    # Natively optimizes for Normalized Discounted Cumulative Gain
        "task": "rank",                      # Instructs FLAML to treat this as Learning-to-Rank
        "estimator_list": ["xgboost"],       # Force it to only tune XGBoost (You can add "lgbm" or "catboost" here too!)
        "log_file_name": "flaml_xgboost.log",
        "verbose": 3                         # Show detailed progress
    }

    # 4. Execute the Frugal Search
    print("Starting Cost-Frugal Search. This will test small models first and scale up...")
    automl.fit(
        X_train=X_train, y_train=y_train, groups=groups_train,
        X_val=X_val, y_val=y_val, groups_val=groups_val,
        **flaml_settings
    )

    # 5. Extract and format the results
    print("\n=== FLAML TUNING COMPLETE ===")
    best_config = automl.best_config
    
    print(f"Best Validation NDCG: {1.0 - automl.best_loss:.5f}") # FLAML minimizes loss, so 1 - loss = score
    print("Best Parameters:")
    for key, value in best_config.items():
        print(f"    '{key}': {value},")
        
    with open('flaml_best_xgb_params.json', 'w') as f:
        json.dump(best_config, f, indent=4)
        
    return best_config

if __name__ == "__main__":
    # Run it for 1 hour (3600 seconds) on 50% of the data
    fast_tune_xgboost_with_flaml(sample_size=1, time_budget_seconds=600)