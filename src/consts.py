from pathlib import Path

SRC_DIR = Path(__file__).parent

PARAMS = {
    'lambdamart': {
        "objective"         : 'rank_xendcg',
        "metric"            : 'ndcg',
        "boosting"          : 'dart',    

        "log_max_bin"       : 10,
        "n_estimators"      : 1464,
        "num_leaves"        : 290,
        "max_depth"         : 11,
        "min_child_samples" : 13,

        "learning_rate"     : 0.014938874966439877,
        "min_gain_to_split" : 0.2728785074451587,
        "reg_alpha"         : 0.002159043586101968,
        "reg_lambda"        : 6.354638231502303,
        
        "subsample"         : 0.5,
        "colsample_bytree"  : 1.0,
        
        "bagging_freq"      : 1,           

        'lambdarank_truncation_level': 8,
        'lambdarank_norm': True,
        "sigmoid": 2.822710794521999,
        
        'extra_trees': False,
        "max_bin": 255,
        "path_smooth": 24.748589600471746
    },
    'xgboost': {
        'n_estimators': 2052,
        'max_leaves': 511,
        'min_child_weight': 24.554806906208903,
        'learning_rate': 0.02058763133924334,
        'subsample': 0.991983283208459,
        'colsample_bylevel': 1.0,
        'colsample_bytree': 0.4362147460980912,
        'reg_alpha': 0.0019118378194259021,
        'reg_lambda': 427.7164551309011,
    },
    'ensemble_weights': {
        'xgb': 0.67,
        'lgbm': 0.33
    }
}
