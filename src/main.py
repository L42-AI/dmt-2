import time

from dotenv import load_dotenv
load_dotenv()
from pipeline import Pipeline
from logistics import init_log, record_metrics
from data import export_submission

parameters = {
    'lambdamart': {
        'objective'         : 'lambdarank',
        'metric'            : 'ndcg',
        'eval_at'           : [5],
        'label_gain'        : [0, 1, 0, 0, 0, 5], 
        'num_leaves'        : 212,
        'learning_rate'     : 0.03474073710753371,
        'max_depth'         : 13,
        'min_data_in_leaf'  : 7,
        'feature_fraction'  : 0.6887609293602818,
        'bagging_fraction'  : 0.9419809880561872,
        'bagging_freq'      :  1,
        'lambda_l1'         : 2.6412275461924535e-05,
        'lambda_l2'         : 9.944461721519527,
        'min_gain_to_split' : 0.8895511368523484,
        'path_smooth'       : 0.5236868356060049,
        'tree_learner'      : 'voting',
        'lambdarank_position_bias_regularization' : 5,
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

start_time = time.monotonic()
pipeline = Pipeline(parameters = parameters, sample_size = 0.2, view_importance = True)

# Setup approaches and metrics (to be shown in logs)
approaches = ['xgboost', 'lambdamart', 'ensemble']
approaches = ['lambdamart']
metric_names = ['Training Accuracy', 'Validation Accuracy', 'Test Accuracy']

# Write header
init_log(metric_names)

for approach in approaches:
    train_predictions, val_predictions, test_predictions, train_acc, val_acc, test_acc = pipeline.run(approach)
    # export_submission(test_predictions, True)

    # Terminal Output
    print(f'=== {approach} ===')
    print(f"Train Accuracy: {train_acc}")
    print(f"Validation Accuracy: {val_acc}")
    print(f"Test Accuracy: {test_acc}")
    print('\n')

    # Record metrics
    metrics = [train_acc, val_acc, test_acc]
    record_metrics(approach, metrics)

end_time = time.monotonic()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
