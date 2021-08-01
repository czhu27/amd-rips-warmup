import os

import tensorflow as tf
from tensorflow import keras
import numpy as np

from plots import make_movie
from data import compute_error_wave

# Load in already trained model to test with different inputs

test_data_dir = "data/wave/05-025"
# Get the latest timestamp
subpaths = os.listdir(test_data_dir)
assert len(subpaths) == 1, "Must have exactly one data timestamp"
test_data_dir = test_data_dir + "/" + subpaths[-1]
fpath = test_data_dir + '/' + 'processed_data.npz'
assert os.path.exists(fpath)
test_data = np.load(fpath)
int_test = test_data['int_test']
ext_test = test_data['ext_test']

test_source = [0.25, 0.5]
model_path = "output/wave/single/test/run_20210730-143514"
output_dir = model_path + "/change_sources/" + str(test_source[0]) + str(test_source[1])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_path = model_path + "/trial_0/model"


model = keras.models.load_model(model_path)
make_movie(model, output_dir, test_source=test_source)
print(compute_error_wave(model, int_test, source_input=test_source))