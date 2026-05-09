import json
import argparse
from src.data.loader import load_data
from src.features.build_features import process_features
from src.models.train import train_model
from src.evaluation.metrics import evaluate

def run_pipeline(config_path):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 1. Load Data
    print("Loading data...")
    # data = load_data(config['data_path'])
    
    # 2. Process Features and Imputation
    print(f"Processing features using {config['imputation_method']} imputation...")
    # processed_data = process_features(data, config)
    
    # 3. Train Model
    print(f"Training {config['model']}...")
    # model = train_model(processed_data, config)
    
    # 4. Evaluate
    print("Evaluating model...")
    # metrics = evaluate(model, processed_data)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recommender pipeline")
    parser.add_argument("--config", type=str, default="configs/experiment_config.json", help="Path to experiment config JSON")
    args = parser.parse_args()
    
    run_pipeline(args.config)
