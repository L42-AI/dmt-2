from pathlib import Path

import pandas as pd
import kagglehub

def load_competition_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Returns submission_sample, training_set, test_set dataframes from the competition dataset. """
    path = Path(kagglehub.competition_download('dmt-2026-2nd-assignment'))
    return (
        pd.read_csv(path / 'submission_sample.csv'),
        pd.read_csv(path / 'training_set_VU_DM.csv'),
        pd.read_csv(path / 'test_set_VU_DM.csv')
    )