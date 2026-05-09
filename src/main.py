import sys
from dotenv import load_dotenv
load_dotenv()

from kaggle_interaction import export_submission, load_test_set
import predict as p
import data as d

data = load_test_set()

data = d.scale_bounded(data)
data = d.compute_comp_rates(data)

data = p.random(data)

export_submission(data, push_to_kaggle=False)