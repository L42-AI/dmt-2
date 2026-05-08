from dotenv import load_dotenv
load_dotenv()

from kaggle_interaction import export_submission, load_test_set
import predict

predictions = predict.random(load_test_set())
export_submission(predictions, push_to_kaggle=True)