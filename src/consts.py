from pathlib import Path

SRC_DIR = Path(__file__).parent

PARAMS = {
    'lambdamart': {
        "objective"         : 'rank_xendcg',
        "metric"            : 'ndcg',
        "boosting"          : 'dart',    

        "n_estimators"      : 639,
        "num_leaves"        : 225,
        "max_depth"         : 9,
        "min_child_samples" : 7,

        "learning_rate"     : 0.059024870262997915,
        "min_gain_to_split" : 0.02230882376865448,
        "reg_alpha"         : 0.09130717779270929,
        "reg_lambda"        : 1000.0,
        
        "subsample"         : 0.8243444512755314,
        "colsample_bytree"  : 0.8392520450895368,
        
        "bagging_freq"      : 1,           
        
        "label_gain"        : [0, 1, 0, 0, 0, 31], 

        'lambdarank_truncation_level': 8,
        'lambdarank_norm': True,
        'sigmoid': 1.2,
        
        'extra_trees': True,
        'max_bin': 127,
        "path_smooth"       : 0.54,
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
