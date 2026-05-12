import pandas as pd

from pathlib import Path

def export_submission(data: pd.DataFrame, push_to_kaggle=False):
    """
    Exports predictions to a CSV and optionally submits directly to Kaggle.
    """

    file = Path(__file__).parent / 'submission.csv'
    comp_id='dmt-2026-2nd-assignment'
    message='Pipeline submission from DMT-2 assignment'

    submission_df = data[['srch_id', 'prop_id', 'position']]
    submission_df = submission_df.sort_values(['srch_id', 'position']).drop(columns=['position'])

    submission_df.to_csv(file, index=False)

    print(f"Successfully exported {len(submission_df)} rows to '{file}'.")

    if push_to_kaggle:
        import kaggle
        print(f"Submitting '{file.name}' to Kaggle competition: {comp_id}...")
        try:
            kaggle.api.competition_submit(str(file), message, comp_id)
            print("Successfully pushed to Kaggle!")
        except Exception as e:
            print(f"Upload error: {e}")
            print("Ensure your kaggle.json API token is correctly placed in your ~/.kaggle/ directory.")