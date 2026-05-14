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
    }\
}
pipeline = Pipeline(parameters = parameters, sample_size = 1)

# Setup approaches and metrics (to be shown in logs)
# approaches = ['ceiling', 'baseline', 'lambdamart', 'xgboost']
approaches = ['xgboost']
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

