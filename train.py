import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import os
import io
import datetime
import argparse
import copy
from tensorflow.python.ops.gen_array_ops import size

import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

from helpers import Configs
from nn import create_nn
from targets import get_target

from toy_data import data_creation

#tf.debugging.set_log_device_placement(True)

def get_data(configs):
    # Data for training NN based on L_f loss function
	X_l, X_ul = data_creation(configs.dataset, configs.corners)

	# Set target function
	f, grad_reg = get_target(configs.target, configs.gradient_loss, configs)

	Y_l = f(X_l[:, 0:1], X_l[:, 1:2])

	return X_l, Y_l, X_ul, grad_reg

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
	X_l, Y_l, X_ul, grad_reg = get_data(configs)

	Y_ul = tf.zeros((X_ul.shape[0], 1))

	if configs.noise > 0:
		Y_l += np.reshape(configs.noise*np.random.randn((len(Y_l))),(len(Y_l),1))
		Y_ul += np.reshape(configs.noise*np.random.randn((len(Y_ul))),(len(Y_ul),1))

	is_labeled_l = tf.fill(Y_l.shape, True)
	is_labeled_ul = tf.fill(Y_ul.shape, False)

	X_all = tf.concat([X_l, X_ul], axis=0)
	Y_all = tf.concat([Y_l, Y_ul], axis=0)
	is_labeled_all = tf.concat([is_labeled_l, is_labeled_ul], axis=0)

	# if "data-distribution" in configs.plots:
	# 	print("Saving data distribution plots")
	# 	plot_data(X_l, "labeled", figs_folder)
	# 	plot_data(X_ul, "unlabeled", figs_folder)
	# 	plot_data(X_all, "all", figs_folder)
	

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
	# Assess accuracy with non-optimized model
	# ------------------------------------------------------------------------------
	# f_pred_0 = model.predict(X_l)
	# error_0 = np.sqrt(np.mean(np.square(f_pred_0 - Y_l)))

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
				# Make grid to display true function and predicted
				error1 = compute_error(model, f, -1.0, 1.0)
				tf.summary.scalar('Error/interpolation', data=error1, step=epoch)
				error2 = compute_error(model, f, -2.0, 2.0)
				tf.summary.scalar('Error/extrapolation', data=error2, step=epoch)

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
	f_pred_1 = model.predict(X_l)
	error_1 = np.sqrt(np.mean(np.square(f_pred_1 - Y_l)))
	loss_value = model.loss_function_f(f_pred_1, Y_l)/X_l.shape[0]

	print("Train set error (before opt): {:.15E}".format(error_0))
	print("Train set error (after opt) : {:.15E}".format(error_1))
	print("Ratio of errors             : {:.1F}".format(error_0/error_1))
	print("Loss function value         : {:.15E}".format(loss_value))

	# ------------------------------------------------------------------------------
	# Stress set - Assess extrapolation capabilities
	# ------------------------------------------------------------------------------

	# Make grid to display true function and predicted
	error1 = compute_error(model, f, -1.0, 1.0)
	print("Error [-1,1]x[-1,1] OLD: {:.6E}".format(error1))
	#error2 = compute_error(model, f, -2.0, 2.0)
	error2 = extrap_error(model, f, -1.0, 1.0, -2.0, 2.0)
	print("Error [-2,2]x[-2,2]: {:.6E}".format(error2))
	#error3 = compute_error(model, f, -3.0, 3.0)
	error3 = extrap_error(model, f, -2.0, 2.0, -3.0, 3.0)
	print("Error [-3,3]x[-3,3]: {:.6E}".format(error3))

	if "extrapolation" in configs.plots:
		print("Saving extrapolation plots")
		buf = plot_gridded_functions(model, f, -1.0, 1.0, "100", folder=figs_folder)
		buf = plot_gridded_functions(model, f, -2.0, 2.0, "200", folder=figs_folder)
		buf = plot_gridded_functions(model, f, -3.0, 3.0, "300", folder=figs_folder)

	os.makedirs(results_dir, exist_ok=True)
	with open(results_dir + '/results.yaml', 'w') as outfile:
		e1, e2, e3, l1 = (float("{:.6E}".format(error1)), float("{:.6E}".format(error2)), 
			float("{:.6E}".format(error3)), float("{:.6E}".format(loss_value)))
		trainTime = "{:.2F} s".format(toc - tic)
		yaml.dump({'error1': e1, 'error2': e2, 'error3': e3, 'loss_value': l1,
		'training_time': trainTime}, outfile, default_flow_style=False)