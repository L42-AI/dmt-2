import kaggle

import pandas as pd

def export_submission(
    test_ids,
    predictions, 
    id_col='srch_id', 
    target_col='prop_id', 
    filename='submission.csv',
    push_to_kaggle=False,
    comp_id='dmt-2026-2nd-assignment',
    message='Automated pipeline submission'
):
    """
    Exports predictions to a CSV and optionally submits directly to Kaggle.
    """
    
    # 1. Verify lengths match
    if len(test_ids) != len(predictions):
        raise ValueError(f"Length mismatch: {len(test_ids)} IDs vs {len(predictions)} predictions.")
    
    # 2. Construct and save the DataFrame
    submission_df = pd.DataFrame({
        id_col: test_ids,
        target_col: predictions
    })
    
    submission_df.to_csv(filename, index=False)
    print(f"Successfully exported {len(submission_df)} rows to '{filename}'.")
    
    # 3. Handle optional Kaggle push
    if push_to_kaggle:
        if kaggle is None:
            print("Upload failed: The 'kaggle' Python package is not installed.")
            print("Your CSV was saved locally. Install it via 'pip install kaggle' to upload.")
            return
            
        print(f"Submitting '{filename}' to Kaggle competition: {comp_id}...")
        try:
            kaggle.api.competition_submit(filename, message, comp_id)
            print("Successfully pushed to Kaggle!")
        except Exception as e:
            print(f"Upload error: {e}")
            print("Ensure your kaggle.json API token is correctly placed in your ~/.kaggle/ directory.")