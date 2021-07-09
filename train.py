import numpy as np
import time
import os

import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

from helpers import Configs
from nn import create_nn
from targets import get_target
from plots import plot_data_2D, plot_gridded_functions
from data import data_creation, compute_error, extrap_error


#tf.debugging.set_log_device_placement(True)

def get_data(configs):

	if configs.source == "synthetic":

		# Data for training NN based on L_f loss function
		X_l, X_ul = data_creation(configs.dataset, configs.corners)

		# Set target function
		f, grad_reg = get_target(configs.target, configs.gradient_loss, configs)

		# Apply target func to data
		Y_l = f(X_l[:, 0:1], X_l[:, 1:2])

		error_metrics = {
			"interpolation error (1x1 square)": lambda model : compute_error(model, f, -1.0, 1.0),
			"extrapolation error (2x2 ring)": lambda model : extrap_error(model, f, -1.0, 1.0, -2.0, 2.0),
			"extrapolation error (3x3 ring)": lambda model : extrap_error(model, f, -2.0, 2.0, -3.0, 3.0),
		}

	elif configs.source == "wave":
		raise ValueError("THIS IS WHERE WAVE EQ DATA IS LOADED")
	else:
		raise ValueError("Unknown data source " + configs.source)

	return X_l, Y_l, X_ul, grad_reg, error_metrics


def plot_data(X_l, X_ul, figs_folder, configs):
	
	if X_l.shape[1] == 2:
		# 2D Plotting
		plot_data_2D(X_l, X_ul, figs_folder)
	elif X_l.shape[1] == 3:
		# TODO: 3d plots here
		for i in range(7):
			print("PLOT PLOT PLOT")

def comparison_plots(model, figs_folder, configs):

	if configs.source == "synthetic":
		# 2D Plotting

		# Set target function
		f, grad_reg = get_target(configs.target, configs.gradient_loss, configs)

		print("Saving extrapolation plots")
		buf = plot_gridded_functions(model, f, -1.0, 1.0, "100", folder=figs_folder)
		buf = plot_gridded_functions(model, f, -2.0, 2.0, "200", folder=figs_folder)
		buf = plot_gridded_functions(model, f, -3.0, 3.0, "300", folder=figs_folder)

	elif configs.source == "wave":
		# 3D Plotting

		print("3D MOVIE HERE")
	
	else:
		raise ValueError("Unknown data source " + configs.source)


def train(configs: Configs):

	# ------------------------------------------------------------------------------
	# General setup
	# ------------------------------------------------------------------------------

	# Set device (GPU, CPU)
	if configs.device == "gpu":
		os.environ["CUDA_VISIBLE_DEVICES"]= "0"
	elif configs.device == "cpu":
		os.environ["CUDA_VISIBLE_DEVICES"]= ""
	else:
		raise ValueError("Unknown device " + configs.device)

	# Setup folder structure vars
	output_dir = configs.output_dir
	# TB logs
	log_dir = output_dir + "/logs" 
	# TODO: This is a hack. Understand tensorboard dirs better...
	scalar_dir = log_dir + "/train" #+ "/scalars"
	metrics_dir = scalar_dir #+ "/metrics"
	figs_folder = output_dir + "/figs"
	results_dir = output_dir + "/results"
	os.makedirs(figs_folder, exist_ok=True)
	os.makedirs(output_dir, exist_ok=True)
	print("Saving to output dir: ", output_dir)

	# Save configs
	yaml.safe_dump(configs.__dict__, open(output_dir + "/configs.yaml", "w"))

	# Setup Tensorboard
	file_writer = tf.summary.create_file_writer(metrics_dir)
	file_writer.set_as_default()

	# Set seeds for reproducibility
	np.random.seed(configs.seed)
	tf.random.set_seed(configs.seed)

	# ------------------------------------------------------------------------------
	# Data preparation
	# ------------------------------------------------------------------------------

	# Interesting		-- Boring
	# X_l, Y_l, X_ul 	-- Y_ul, is_labeled

	# Get the "interesting" data
	X_l, Y_l, X_ul, grad_reg, error_metrics = get_data(configs)

	Y_ul = tf.zeros((X_ul.shape[0], 1))

	# Add noise to y-values
	if configs.noise > 0:
		Y_l += tf.random.normal(Y_l.shape, stddev=configs.noise)
		#Y_l += np.reshape(configs.noise*np.random.randn((len(Y_l))),(len(Y_l),1))

	is_labeled_l = tf.fill(Y_l.shape, True)
	is_labeled_ul = tf.fill(Y_ul.shape, False)

	X_all = tf.concat([X_l, X_ul], axis=0)
	Y_all = tf.concat([Y_l, Y_ul], axis=0)
	is_labeled_all = tf.concat([is_labeled_l, is_labeled_ul], axis=0)

	if "data-distribution" in configs.plots:
		plot_data(X_l, X_ul, figs_folder, configs)

	# Create TensorFlow dataset for passing to 'fit' function (below)
	dataset = tf.data.Dataset.from_tensors((X_all, Y_all, is_labeled_all))

	# ------------------------------------------------------------------------------
	# Create neural network (physics-inspired)
	# ------------------------------------------------------------------------------
	layers = configs.layers
	model = create_nn(layers, configs)
	model.summary()

	# TODO: Hacky add...
	model.gradient_regularizer = grad_reg

	# ------------------------------------------------------------------------------
	# Model compilation / training (optimization)
	# ------------------------------------------------------------------------------
	if configs.lr_scheduler:
		opt_step = tf.keras.optimizers.schedules.PolynomialDecay(
			configs.lr_scheduler_params[0], configs.lr_scheduler_params[2], 
			end_learning_rate=configs.lr_scheduler_params[1], power=configs.lr_scheduler_params[3],
			cycle=False, name=None) #Changing learning rate
	else:
		print(type(configs.lr))
		if not isinstance(configs.lr, float):
			raise ValueError("configs.lr must be floats (missing a decimal point?)")
		opt_step = configs.lr		# gradient descent step

	opt_batch_size = configs.batch_size	# batch size
	opt_num_its = configs.epochs		# number of iterations

	model.set_batch_size(opt_batch_size)

	optimizer = optimizers.Adam(learning_rate = opt_step)
	model.compile(optimizer = optimizer, run_eagerly=configs.debug)		# DEBUG
	tic = time.time()

	# Define Tensorboard Callbacks
	class TimeLogger(keras.callbacks.Callback):
		def __init__(self):
			pass
		def on_train_begin(self, logs):
			self.train_start = time.time()
		def on_epoch_begin(self, epoch, logs):
			self.epoch_start = time.time()
		def on_epoch_end(self, epoch, logs=None):
			train_dur = time.time() - self.train_start
			epoch_dur = time.time() - self.epoch_start
			tf.summary.scalar('Time/Total', data=train_dur, step=epoch)
			tf.summary.scalar('Time/Epoch', data=epoch_dur, step=epoch)

	class StressTestLogger(keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs):
			self.test_every = 100
			if epoch % self.test_every == self.test_every - 10:
				for error_name, error_func in error_metrics.items():
					error_val = error_func(model)
					tf.summary.scalar('Error/' + error_name, data=error_val, step=epoch)

	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	logging_callbacks = [TimeLogger(), StressTestLogger(), tensorboard_callback]

	if "tensorboard" in configs.plots:
		print("Using tensorboard callbacks")
		callbacks = logging_callbacks
	else:
		callbacks = []

	model.fit(dataset, 
			epochs=opt_num_its, 
			verbose=2,
			callbacks=callbacks)
	toc = time.time()
	print("Training time: {:.2F} s\n".format(toc - tic))

	if "model" in configs.saves:
		print("Saving final model")
		model.save(output_dir + "/model")

	# ------------------------------------------------------------------------------
	# Assess accuracy with optimized model and compare with non-optimized model
	# ------------------------------------------------------------------------------
	# ------------------------------------------------------------------------------
	# Stress set - Assess extrapolation capabilities
	# ------------------------------------------------------------------------------
	final_metrics = {}

	Y_pred_l = model.predict(X_l)
	loss_value = model.loss_function_f(Y_pred_l, Y_l)/X_l.shape[0]
	print("FINAL Loss: \t\t\t {:.6E}".format(loss_value))
	final_metrics['loss_value'] = "{:.6E}".format(loss_value)

	for error_name, error_func in error_metrics.items():
		error_val = error_func(model)
		print('FINAL Error/' + error_name + f":\t\t\t {error_val:.6E}")
		final_metrics[error_name] = "{:.6E}".format(error_val)

	if "extrapolation" in configs.plots:
		comparison_plots(model, figs_folder, configs)

	train_time = toc - tic 
	final_metrics['training_time'] = "{:.2F} s".format(train_time)

	os.makedirs(results_dir, exist_ok=True)
	with open(results_dir + '/results.yaml', 'w') as outfile:
		yaml.dump(final_metrics, outfile, default_flow_style=False)