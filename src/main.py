from dotenv import load_dotenv
load_dotenv()
from pipeline import Pipeline
from logistics import init_log, record_metrics

parameters = {
    'lambdamart': {
        'num_leaves'        : 31,
        'learning_rate'     : 0.5,
        'n_estimators'      : 1000,
        'min_data_in_leaf':100,   # minimum samples per leaf
        'max_depth':6,            # limit tree depth
        'learning_rate':0.05,     # slower learning
        'reg_alpha':0.1,          # L1 regularisation
        'reg_lambda':0.1,         # L2 regularisation
        'subsample':0.8,          # row subsampling per tree
        'colsample_bytree':0.8    # feature subsampling per tree
    }
}
pipeline = Pipeline(parameters = parameters, sample_size = 0.1)

# Setup approaches and metrics (to be shown in logs)
# approaches = ['ceiling', 'baseline', 'lambdamart', 'xgboost']
approaches = ['lambdamart', 'xgboost']
metric_names = ['Training Accuracy', 'Validation Accuracy', 'Test Accuracy']

# Write header
init_log(metric_names)

for approach in approaches:
    train_predictions, val_predictions, test_predictions, train_acc, val_acc, test_acc = pipeline.run(approach)

    # Terminal Output
    print(f'=== {approach} ===')
    print(f"Train Accuracy: {train_acc}")
    print(f"Validation Accuracy: {val_acc}")
    print(f"Test Accuracy: {test_acc}")
    print('\n')

    # Record metrics
    metrics = [train_acc, val_acc, test_acc]
    record_metrics(approach, metrics)

