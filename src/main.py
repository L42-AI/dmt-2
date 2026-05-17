import time

from dotenv import load_dotenv
load_dotenv()
from pipeline import Pipeline
from logistics import init_log, record_metrics
from data import export_submission

from consts import PARAMS

start_time = time.monotonic()
pipeline = Pipeline(parameters = PARAMS, sample_size = .1, view_importance = True)

# Setup approaches and metrics (to be shown in logs)
approaches = ['xgboost', 'lambdamart', 'ensemble']
approaches = ['lambdamart']
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
