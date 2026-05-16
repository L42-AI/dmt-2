import json

from flaml import AutoML, tune

from data import load_data, preprocess_data

def exhaustive_multi_tune_with_flaml(sample_size: float = 1, total_time_budget: int = 18000):
    models = ["lgbm", "xgboost"]
    time_per_model = total_time_budget // len(models)
    
    print(f"=== INITIALIZING FLAML SEQUENTIAL TUNER ===")
    print(f"Total Budget: {total_time_budget}s | Allocating {time_per_model}s per model")
    
    # 1. Load and prep data
    train_set, test_set = load_data(sample_size, random_state=42)
    train_df, val_df, _ = preprocess_data(train_set, test_set)
    
    train_df = train_df.sort_values('srch_id')
    val_df = val_df.sort_values('srch_id')
    
    drop_cols = ['relevance', 'position', 'click_bool', 'booking_bool', 'srch_id', 'prop_id', 'date_time', 'checkin_date', 'checkout_date']
    
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['relevance']
    groups_train = train_df.groupby('srch_id').size().values
    
    X_val = val_df.drop(columns=drop_cols, errors='ignore')
    y_val = val_df['relevance']
    groups_val = val_df.groupby('srch_id').size().values

    # 2. Define the Ultimate Custom Search Spaces
    custom_search_space = {
        "xgboost": {
            "n_estimators": {"domain": tune.lograndint(100, 2000)},
            "max_leaves": {"domain": tune.lograndint(15, 1024)}, 
            "max_depth": {"domain": tune.randint(4, 12)},            
            "min_child_weight": {"domain": tune.loguniform(1, 100)},
            "learning_rate": {"domain": tune.loguniform(0.005, 0.2)},
            "subsample": {"domain": tune.uniform(0.5, 1.0)},
            "colsample_bytree": {"domain": tune.uniform(0.4, 1.0)},
            "colsample_bylevel": {"domain": tune.uniform(0.4, 1.0)},
            "gamma": {"domain": tune.loguniform(0.01, 20.0)},        
            "reg_alpha": {"domain": tune.loguniform(0.001, 100.0)},  
            "reg_lambda": {"domain": tune.loguniform(0.001, 1000.0)} 
        },
        "lgbm": {
            "n_estimators": {"domain": tune.lograndint(100, 2000)},
            "num_leaves": {"domain": tune.lograndint(15, 1024)},
            "max_depth": {"domain": tune.randint(4, 12)},            
            "min_child_samples": {"domain": tune.lograndint(2, 100)},
            "learning_rate": {"domain": tune.loguniform(0.005, 0.2)},
            "subsample": {"domain": tune.uniform(0.5, 1.0)},
            "colsample_bytree": {"domain": tune.uniform(0.4, 1.0)},
            "min_gain_to_split": {"domain": tune.loguniform(0.01, 20.0)}, 
            "reg_alpha": {"domain": tune.loguniform(0.001, 100.0)},  
            "reg_lambda": {"domain": tune.loguniform(0.001, 1000.0)} 
        }
    }

    best_configs = {}

    # 3. Sequential Execution Loop
    for model_name in models:
        print("\n" + "="*50)
        print(f"🚀 NOW TUNING: {model_name.upper()} (Budget: {time_per_model}s)")
        print("="*50)
        
        automl = AutoML()
        
        flaml_settings = {
            "time_budget": time_per_model,  
            "metric": "ndcg@5",                  
            "task": "rank",                      
            "estimator_list": [model_name], # Isolate the single model
            "custom_hp": {model_name: custom_search_space[model_name]}, # Pass only its specific grid
            "log_file_name": f"flaml_{model_name}.log",
            "verbose": 3                         
        }

        # Run the search
        automl.fit(
            X_train=X_train, y_train=y_train, groups=groups_train,
            X_val=X_val, y_val=y_val, groups_val=groups_val,
            **flaml_settings
        )

        # Extract and save
        config = automl.best_config
        best_configs[model_name] = config
        
        print(f"\n✅ {model_name.upper()} TUNING COMPLETE")
        print(f"Best Validation NDCG@5: {1.0 - automl.best_loss:.5f}")
        
        filename = f'flaml_best_{model_name}_params.json'
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved optimal {model_name.upper()} parameters to '{filename}'")

    print("\n=== ALL MODELS TUNED SUCCESSFULLY ===")
    return best_configs

if __name__ == "__main__":
    # 7200 seconds total = 1 hour for LightGBM, 1 hour for XGBoost
    exhaustive_multi_tune_with_flaml(sample_size=1, total_time_budget=3600 * 5)