import time

from dotenv import load_dotenv
load_dotenv()
from pipeline import Pipeline
from logistics import init_log, record_metrics
from data import export_submission

parameters = {
    'lambdamart': {
        'num_leaves'        : 31,
        'learning_rate'     : 0.1,
        'n_estimators'      : 100
    },
    'xgboost': {
        # 'max_depth'         : 7,
        # 'learning_rate'     : 0.035542238661166375,
        # 'n_estimators'      : 800,
        # 'subsample'         : 0.6024637141781495,
        # 'colsample_bytree'  : 0.6952921993037987,
        # 'min_child_weight'  : 22,
        # 'gamma'             : 0.5,
        # 'reg_lambda'        : 15.0,
        # 'reg_alpha'         : 0.0
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
        'xgb': 0.65,
        'lgbm': 0.35
    }
}

start_time = time.monotonic()
pipeline = Pipeline(parameters = parameters, sample_size = .2)

# Setup approaches and metrics (to be shown in logs)
approaches = ['xgboost', 'lambdamart', 'ensemble']
# approaches = ['ensemble']
metric_names = ['Train NDCG@5', 'Validation NDCG@5', 'Test NDCG@5']
# Write header
init_log(metric_names)

for approach in approaches:
    train_predictions, val_predictions, test_predictions, train_acc, val_acc, test_acc = pipeline.run(approach)
    # export_submission(test_predictions, True)

    # Terminal Output
    print(f'=== {approach} ===')
    print(f"Train NDCG@5:      {train_acc:.5f}")
    print(f"Validation NDCG@5: {val_acc:.5f}")
    if test_acc is not None:
        print(f"Test NDCG@5:       {test_acc:.5f}")
    print('\n')

    # Record metrics
    metrics = [train_acc, val_acc, test_acc]
    record_metrics(approach, metrics)

end_time = time.monotonic()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")