from dotenv import load_dotenv
load_dotenv()
from pipeline import Pipeline

pipeline = Pipeline()

for approach in ['ceiling', 'baseline', 'lambdamart', 'xgboost']:
    train_predictions, val_predictions, test_predictions, train_acc, val_acc, test_acc = pipeline.run(approach)
    print(f'=== {approach} ===')
    print(f"Train Accuracy: {train_acc}")
    print(f"Validation Accuracy: {val_acc}")
    print(f"Test Accuracy: {test_acc}")
    print('\n')