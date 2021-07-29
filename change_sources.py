import os

import tensorflow as tf
from tensorflow import keras

from plots import make_movie

# Load in already trained model to test with different inputs


test_source = [0.4, 0.7]
model_path = "output/wave/single/test/run_20210728-145844"
output_dir = model_path + "/change_sources/" + str(test_source[0]) + str(test_source[1])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_path = model_path + "/trial_0/model"


model = keras.models.load_model(model_path)
make_movie(model, output_dir, test_source=test_source)