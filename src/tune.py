import argparse
import optuna
import json
from dotenv import load_dotenv

load_dotenv()

from pipeline import Pipeline

# ==========================================
# 1. OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial, sample_size):
    """
    The evaluation function Optuna runs for every trial.
    """
    # 1. Define the XGBoost Search Space
    xgb_params = {
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50)
    }
    
    # We leave LambdaMART static since we are only tuning XGBoost right now
    dynamic_parameters = {
        'lambdamart': {
            'num_leaves'        : 31,
            'learning_rate'     : 0.1,
            'n_estimators'      : 100
        },
        'xgboost': xgb_params
    }
    
    # 2. Initialize and Run the Pipeline
    pipeline = Pipeline(parameters=dynamic_parameters, sample_size=sample_size)
    
    # We only care about the Validation Accuracy (NDCG) for the XGBoost approach
    _, _, _, _, val_acc, _ = pipeline.run('xgboost')
    
    return val_acc

# ==========================================
# 2. ARGPARSE SETUP
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for XGBoost Ranker")
    
    parser.add_argument('--n_trials', type=int, default=50, 
                        help="Number of tuning trials to run (default: 50)")
    parser.add_argument('--sample_size', type=float, default=0.1, 
                        help="Fraction of data to sample for faster tuning (default: 0.1)")
    parser.add_argument('--study_name', type=str, default='xgboost_ranker_study', 
                        help="Name of the study (used for the database saving)")
    
    return parser.parse_args()

# ==========================================
# 3. MAIN EXECUTION (ONLINE TRACKING)
# ==========================================
if __name__ == "__main__":
    args = parse_args()

    print(f"=== INITIALIZING OPTUNA TUNING ===")
    print(f"Study Name: {args.study_name}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Target Trials: {args.n_trials}")
    
    # Create an SQLite database to save progress "online"
    # This allows you to stop the script and restart it without losing data!
    storage_url = f"sqlite:///{args.study_name}.db"
    
    study = optuna.create_study(
        direction='maximize', 
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True # Resumes from the database if it already exists
    )
    
    try:
        # Run the optimization
        study.optimize(lambda trial: objective(trial, args.sample_size), n_trials=args.n_trials)
    except KeyboardInterrupt:
        # Gracefully handle manual cancellations (Ctrl+C)
        print("\n[!] Tuning manually interrupted. Progress has been saved to the database.")
    
    # Print the final best results
    print("\n=== TUNING COMPLETE ===")
    print(f"Best Validation NDCG: {study.best_value}")
    print("Best Parameters:")
    
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
        
    # Optional: Save the best parameters to a JSON file so you can easily copy them to main.py
    with open('best_xgb_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print("\nBest parameters saved to 'best_xgb_params.json'")