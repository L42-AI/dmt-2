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
    CatBoostRanker
)

# ==========================================
# 1. FIXED PARAMETERS (The Baseline)
# ==========================================
# These are your strongest known parameters. When tuning one model, 
# the others will fall back to these defaults if needed.
FIXED_PARAMETERS = {
    'lambdamart': {
        'objective'         : 'rank_xendcg',
        'metric'            : 'ndcg',
        'eval_at'           : [5],
        'label_gain'        : [0, 1, 0, 0, 0, 5],
        'n_estimators'      : 2000,
        'num_leaves'        : 212,
        'learning_rate'     : 0.03474073710753371,
        'max_depth'         : 13,
        'min_data_in_leaf'  : 7,
        'feature_fraction'  : 0.6887609293602818,
        'bagging_fraction'  : 0.9419809880561872,
        'bagging_freq'      : 1,
        'lambda_l1'         : 2.6412275461924535e-05,
        'lambda_l2'         : 9.944461721519527,
        'min_gain_to_split' : 0.8895511368523484,
        'path_smooth'       : 0.5236868356060049,
        'tree_learner'      : 'voting',
        'max_bin'           : 127,
        'verbose'           : -1
    },
    'xgboost': {
        'max_depth'         : 7,
        'learning_rate'     : 0.035542238661166375,
        'n_estimators'      : 800,
        'subsample'         : 0.6024637141781495,
        'colsample_bytree'  : 0.6952921993037987,
        'min_child_weight'  : 22,
        'gamma'             : 0.5,
        'reg_lambda'        : 15.0,
        'reg_alpha'         : 0.0
    },
    'catboost': {
        'iterations'        : 1500,
        'learning_rate'     : 0.03,
        'depth'             : 6,
        'l2_leaf_reg'       : 3.0,
        'cat_features': [
            'prop_id', 
            'srch_destination_id', 
            'prop_country_id', 
            'visitor_location_country_id'
        ]
    }
}

# ==========================================
# 2. INDIVIDUAL MODEL OPTUNA OBJECTIVES
# ==========================================

def tune_xgboost(trial, sample_size):
    """Tunes XGBoost Base Model"""
    xgb_params = {
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 50.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True) # Lasso Feature Selection
    }
    
    params = FIXED_PARAMETERS.copy()
    params['xgboost'] = xgb_params
    
    pipeline = Pipeline(parameters=params, sample_size=sample_size)
    _, _, _, _, val_acc, _ = pipeline.run('xgboost')
    return val_acc


def tune_lambdamart(trial, sample_size):
    """Tunes LightGBM LambdaMART with rank_xendcg objective.

    Fixed params (not tuned):
      - objective / metric / label_gain  : define the ranking task, must not change
      - n_estimators + early_stopping(50): effective tree count is decided by early stopping
      - tree_learner='voting' + max_bin=127: memory-efficient settings for full-data runs
      - bagging_freq=1                   : required for bagging_fraction to take effect
    """
    lgbm_params = {
        # --- Fixed ---
        'objective'         : 'rank_xendcg',
        'metric'            : 'ndcg',
        'eval_at'           : [5],
        'label_gain'        : [0, 1, 0, 0, 0, 5],
        'n_estimators'      : 2000,
        'bagging_freq'      : 1,
        'tree_learner'      : 'voting',
        'max_bin'           : 127,
        'verbose'           : -1,
        # --- Tuned ---
        'num_leaves'        : trial.suggest_int('num_leaves', 63, 511),
        'learning_rate'     : trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'max_depth'         : trial.suggest_int('max_depth', 5, 15),
        'min_data_in_leaf'  : trial.suggest_int('min_data_in_leaf', 5, 100),
        'feature_fraction'  : trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction'  : trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'lambda_l1'         : trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2'         : trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'path_smooth'       : trial.suggest_float('path_smooth', 0.0, 2.0),
        'min_gain_to_split' : trial.suggest_float('min_gain_to_split', 0.0, 2.0),
    }

    params = FIXED_PARAMETERS.copy()
    params['lambdamart'] = lgbm_params

    pipeline = Pipeline(parameters=params, sample_size=sample_size)
    _, _, _, _, val_acc, _ = pipeline.run('lambdamart')
    return val_acc


def tune_catboost(trial, sample_size):
    """Tunes CatBoost Base Model"""
    cat_params = {
        'iterations': trial.suggest_int('iterations', 500, 2000, step=250),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
        'cat_features': FIXED_PARAMETERS['catboost']['cat_features'] # Keep this static
    }
    
    params = FIXED_PARAMETERS.copy()
    params['catboost'] = cat_params
    
    pipeline = Pipeline(parameters=params, sample_size=sample_size)
    _, _, _, _, val_acc, _ = pipeline.run('catboost')
    return val_acc


# ==========================================
# 3. ENSEMBLE WEIGHT TUNING LOGIC
# ==========================================

def generate_cached_predictions(sample_size: float) -> pd.DataFrame:
    print(f"\n=== TRAINING BASE MODELS & CACHING PREDICTIONS (Sample Size: {sample_size}) ===")
    pipeline = Pipeline(parameters=FIXED_PARAMETERS, sample_size=sample_size)
    
    train_set = pipeline.train_set.drop(columns=['date_time', 'checkin_date', 'checkout_date'])
    val_set = pipeline.val_set.drop(columns=['date_time', 'checkin_date', 'checkout_date'])

    # Initialize
    xgb_ranker = XGBoostRanker(**FIXED_PARAMETERS['xgboost'])
    lgbm_ranker = LambdaMARTRanker(**FIXED_PARAMETERS['lambdamart'])
    cat_ranker = CatBoostRanker(**FIXED_PARAMETERS['catboost'])

    # Train
    print("[1/3] Training XGBoost...")
    xgb_ranker.train(train_set, val_set, feature_list=pipeline.feature_cols)
    print("[2/3] Training LightGBM...")
    lgbm_ranker.train(train_set, val_set, feature_list=pipeline.feature_cols)
    print("[3/3] Training CatBoost...")
    cat_ranker.feature_names = pipeline.feature_cols
    cat_ranker.train(train_set, val_set)

    # Predict & Cache
    print("\nGenerating Validation Predictions...")
    xgb_val = xgb_ranker.predict(val_set)[['srch_id', 'prop_id', 'position']].rename(columns={'position': 'xgb_pos'})
    lgbm_val = lgbm_ranker.predict(val_set)[['srch_id', 'prop_id', 'position']].rename(columns={'position': 'lgbm_pos'})
    cat_val = cat_ranker.predict(val_set)[['srch_id', 'prop_id', 'position']].rename(columns={'position': 'cat_pos'})

    # Merge
    master_val_df = val_set.copy()
    master_val_df = master_val_df.merge(xgb_val, on=['srch_id', 'prop_id'], how='left')
    master_val_df = master_val_df.merge(lgbm_val, on=['srch_id', 'prop_id'], how='left')
    master_val_df = master_val_df.merge(cat_val, on=['srch_id', 'prop_id'], how='left')
    
    print("Caching Complete!")
    return master_val_df

def tune_ensemble_weights(trial, master_val_df: pd.DataFrame):
    """Lightning-fast objective that only tests Reciprocal Rank Fusion weights."""
    df_eval = master_val_df.copy()
    
    raw_w_xgb = trial.suggest_float('w_xgb', 0.01, 1.0)
    raw_w_lgbm = trial.suggest_float('w_lgbm', 0.01, 1.0)
    raw_w_cat = trial.suggest_float('w_cat', 0.01, 1.0)
    
    total_weight = raw_w_xgb + raw_w_lgbm + raw_w_cat
    w_xgb = raw_w_xgb / total_weight
    w_lgbm = raw_w_lgbm / total_weight
    w_cat = raw_w_cat / total_weight
    
    k = 60
    df_eval['rrf_score'] = (
        (w_xgb * (1.0 / (k + df_eval['xgb_pos']))) +
        (w_lgbm * (1.0 / (k + df_eval['lgbm_pos']))) +
        (w_cat * (1.0 / (k + df_eval['cat_pos'])))
    )
    
    df_eval['position'] = df_eval.groupby('srch_id')['rrf_score'].rank(
        method='first', 
        ascending=False
    ).astype('int32')
    
    df_eval = df_eval.sort_values(by=['srch_id', 'position']).reset_index(drop=True)
    return compute_accuracy(df_eval)


# ==========================================
# 4. ARGPARSE & ROUTING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Target Master Tuner")
    
    parser.add_argument('--target', type=str, required=True, 
                        choices=['xgboost', 'lambdamart', 'catboost', 'ensemble_weights'],
                        help="Which model/component do you want to tune?")
    
    parser.add_argument('--n_trials', type=int, default=50, 
                        help="Number of trials to run")
    parser.add_argument('--sample_size', type=float, default=0.2, 
                        help="Fraction of data to sample")
    parser.add_argument('--study_name', type=str, default=None, 
                        help="Name of the study. Defaults to the target name.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Auto-generate study name if not provided
    study_name = args.study_name if args.study_name else f"study_{args.target}"
    
    print(f"\n=== INITIALIZING OPTUNA MASTER TUNER ===")
    print(f"Target: {args.target.upper()}")
    print(f"Study Name: {study_name}")
    print(f"Sample Size: {args.sample_size}")
    print(f"Target Trials: {args.n_trials}")
    
    storage_url = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        direction='maximize', 
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True
    )
    
    try:
        if args.target == 'ensemble_weights':
            # Run the caching phase first, then the blazing fast weight tuner
            master_val_df = generate_cached_predictions(args.sample_size)
            print("\nStarting Lightning-Fast Weight Tuning...")
            study.optimize(lambda trial: tune_ensemble_weights(trial, master_val_df), n_trials=args.n_trials)
        else:
            # Route to the specific base model tuner
            objective_map = {
                'xgboost': tune_xgboost,
                'lambdamart': tune_lambdamart,
                'catboost': tune_catboost
            }
            target_objective = objective_map[args.target]
            study.optimize(lambda trial: target_objective(trial, args.sample_size), n_trials=args.n_trials)
            
    except KeyboardInterrupt:
        print("\n[!] Tuning manually interrupted. Progress saved to database.")
    
    # ==========================================
    # FINAL OUTPUT FORMATTING
    # ==========================================
    print("\n=== TUNING COMPLETE ===")
    print(f"Best Validation NDCG: {study.best_value:.5f}")
    
    if args.target == 'ensemble_weights':
        # Normalize the final weights for readability
        best_raw = study.best_params
        total = sum(best_raw.values())
        final_params = {
            'xgb': best_raw['w_xgb'] / total,
            'lgbm': best_raw['w_lgbm'] / total,
            'cat': best_raw['w_cat'] / total
        }
        print("\nOptimal Normalized Ensemble Weights:")
    else:
        final_params = study.best_params
        print("\nBest Parameters:")
        
    for key, value in final_params.items():
        print(f"    '{key}': {value},")
        
    # Save the output to a specific JSON file based on the target
    filename = f"best_{args.target}_params.json"
    with open(filename, 'w') as f:
        json.dump(final_params, f, indent=4)
        
    print(f"\nBest parameters saved to '{filename}'")