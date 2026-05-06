import dotenv
dotenv.load_dotenv()

from kaggle_interaction import load_data, export_submission

submittion_sample, training_set, test_set = load_data()

export_submission(submittion_sample, push_to_kaggle=True)