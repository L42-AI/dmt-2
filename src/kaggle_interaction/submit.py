import pandas as pd

from pathlib import Path

def export_submission(submission_data: pd.DataFrame | dict[int, list[int]], push_to_kaggle=False):
    """
    Exports predictions to a CSV and optionally submits directly to Kaggle.
    """

    file = Path(__file__).parent / 'submission.csv'
    comp_id='dmt-2026-2nd-assignment'
    message='Pipeline submission from DMT-2 assignment'

    if isinstance(submission_data, pd.DataFrame):
        submission_df = submission_data
    else:
        submission_df = pd.DataFrame([(srch_id, prop_id) for srch_id, prop_ids in submission_data.items() for prop_id in prop_ids], columns=['srch_id', 'prop_id'])

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