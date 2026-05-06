import dotenv
dotenv.load_dotenv()

from data.load import load_competition_data
load_competition_data()