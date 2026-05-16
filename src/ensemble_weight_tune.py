import argparse
import optuna
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from pipeline import Pipeline
from evaluate import compute_accuracy
from predict import (
    XGBoostRanker,
    LambdaMARTRanker,
)

from consts import PARAMS

# ==========================================
# 2. STAGE 1: CACHE PREDICTIONS
# ==========================================
def generate_cached_predictions(sample_size: float) -> pd.DataFrame:
    print(f"\n=== STAGE 1: TRAINING BASE MODELS & CACHING PREDICTIONS (Sample Size: {sample_size}) ===")
    pipeline = Pipeline(parameters=PARAMS, sample_size=sample_size)
    
    train_set = pipeline.train_set.drop(columns=['date_time', 'checkin_date', 'checkout_date'])
    val_set = pipeline.val_set.drop(columns=['date_time', 'checkin_date', 'checkout_date'])

    # Initialize correctly using the parameter dictionary directly
    xgb_ranker = XGBoostRanker(parameters=PARAMS['xgboost'])
    lgbm_ranker = LambdaMARTRanker(parameters=PARAMS['lambdamart'])

    # Train
    print("[1/2] Training XGBoost...")
    xgb_ranker.train(train_set, val_set)
    print("[2/2] Training LightGBM...")
    lgbm_ranker.train(train_set, val_set)

    # Predict & Cache
    print("\nGenerating Validation Predictions...")
    xgb_val = xgb_ranker.predict(val_set)[['srch_id', 'prop_id', 'position']].rename(columns={'position': 'xgb_pos'})
    lgbm_val = lgbm_ranker.predict(val_set)[['srch_id', 'prop_id', 'position']].rename(columns={'position': 'lgbm_pos'})

    # Merge
    master_val_df = val_set.copy()
    master_val_df = master_val_df.merge(xgb_val, on=['srch_id', 'prop_id'], how='left')
    master_val_df = master_val_df.merge(lgbm_val, on=['srch_id', 'prop_id'], how='left')
    
    print("Caching Complete!")
    return master_val_df

# ==========================================
# 3. STAGE 2: OPTUNA WEIGHT TUNER
# ==========================================
def tune_ensemble_weights(trial, master_val_df: pd.DataFrame):
    """Lightning-fast objective that only tests Reciprocal Rank Fusion weights."""
    df_eval = master_val_df.copy()
    
    raw_w_xgb = trial.suggest_float('w_xgb', 0.01, 1.0)
    raw_w_lgbm = trial.suggest_float('w_lgbm', 0.01, 1.0)
    
    total_weight = raw_w_xgb + raw_w_lgbm
    w_xgb = raw_w_xgb / total_weight
    w_lgbm = raw_w_lgbm / total_weight
    
    k = 60
    df_eval['rrf_score'] = (
        (w_xgb * (1.0 / (k + df_eval['xgb_pos']))) +
        (w_lgbm * (1.0 / (k + df_eval['lgbm_pos'])))
    )
    
    df_eval['position'] = df_eval.groupby('srch_id')['rrf_score'].rank(
        method='first', 
        ascending=False
    ).astype('int32')
    
    df_eval = df_eval.sort_values(by=['srch_id', 'position']).reset_index(drop=True)
    return compute_accuracy(df_eval)


# ==========================================
# 4. ARGPARSE SETUP
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Lightning Fast Ensemble Weight Tuner")
    
    parser.add_argument('--n_trials', type=int, default=500, 
                        help="Number of weight combinations to test (default: 500)")
    parser.add_argument('--sample_size', type=float, default=1.0, 
                        help="Fraction of data to sample (default: 1.0 for final tuning)")
    parser.add_argument('--study_name', type=str, default='ensemble_weights_study', 
                        help="Name of the study (used for the database saving)")
    
    return parser.parse_args()


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    
    print(f"\n=== INITIALIZING ENSEMBLE WEIGHT TUNER ===")
    print(f"Study Name: {args.study_name}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Target Trials: {args.n_trials}")
    
    storage_url = f"sqlite:///{args.study_name}.db"
    
    study = optuna.create_study(
        direction='maximize', 
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True
    )
    
    try:
        # Run the caching phase first, then the blazing fast weight tuner
        master_val_df = generate_cached_predictions(args.sample_size)
        print("\nStarting Lightning-Fast Weight Tuning...")
        study.optimize(lambda trial: tune_ensemble_weights(trial, master_val_df), n_trials=args.n_trials)
            
    except KeyboardInterrupt:
        print("\n[!] Tuning manually interrupted. Progress saved to database.")
    
    # ==========================================
    # FINAL OUTPUT FORMATTING
    # ==========================================
    print("\n=== TUNING COMPLETE ===")
    print(f"Best Validation NDCG@5: {study.best_value:.5f}")
    
    # Normalize the final weights for readability
    best_raw = study.best_params
    total = sum(best_raw.values())
    final_weights = {
        'xgb': best_raw['w_xgb'] / total,
        'lgbm': best_raw['w_lgbm'] / total,
        'cat': best_raw['w_cat'] / total
    }
    
    print("\nOptimal Normalized Ensemble Weights:")
    for model, weight in final_weights.items():
        print(f"    '{model}': {weight:.4f},")
        
    # Save the output to a specific JSON file
    filename = "best_ensemble_weights.json"
    with open(filename, 'w') as f:
        json.dump(final_weights, f, indent=4)
        
    print(f"\nBest normalized weights saved to '{filename}'")