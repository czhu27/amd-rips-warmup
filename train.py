import numpy as np
import time
import os
from tensorflow.python.ops.gen_array_ops import zeros_like

import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

from helpers import Configs
from nn import create_nn
from targets import get_target
from plots import plot_data_2D, plot_gridded_functions, make_movie, make_wave_plot, make_heatmap_movie
from data import data_creation, compute_error, extrap_error, data_wave, compute_error_wave, error_time
from wave_reg import get_wave_reg

#tf.debugging.set_log_device_placement(True)
def general_error(model, X, Y):
	Y_pred = model.predict(X)
	Y_diff = Y - Y_pred
	mse = np.sqrt(np.mean(np.square(Y_diff)))
	return mse

def get_data(configs, figs_folder):
	grad_bools = None #Placeholder
	if configs.source == "synthetic":

		# Data for training NN based on L_f loss function
		X_l, X_ul = data_creation(configs.dataset, configs.corners)

		# Set target function
		f, grad_reg = get_target(configs.target, configs.gradient_loss, configs)

		# Apply target func to data
		Y_l = f(X_l[:, 0:1], X_l[:, 1:2])

		if grad_reg == 'zero' or grad_reg == 'const':
			grad_bools = tf.fill(X_l.shape[0] + X_ul.shape[0], True)

		error_metrics = {
			"interpolation error (1x1 square)": lambda model : compute_error(model, f, -1.0, 1.0),
			"extrapolation error (2x2 ring)": lambda model : extrap_error(model, f, -1.0, 1.0, -2.0, 2.0),
			"extrapolation error (3x3 ring)": lambda model : extrap_error(model, f, -2.0, 2.0, -3.0, 3.0),
		}

	elif configs.source == "wave":
		data_run = configs.data_dir
		data_run = data_run + "/" + configs.data_run
		# Get the latest timestamp
		subpaths = os.listdir(data_run)
		assert len(subpaths) == 1, "Must have exactly one data timestamp"
		data_run = data_run + "/" + subpaths[-1]

		fpath = data_run + '/' + 'processed_data.npz'
		assert os.path.exists(fpath)
		data = np.load(fpath)

		int_label, int_unlabel, bound, int_test = data['int_label'], data['int_unlabel'], data['bound'], data['int_test']
		ext_label, ext_unlabel, ext_test = data['ext_label'], data['ext_unlabel'], data['ext_test']
		X_l = np.float32(np.concatenate((bound[:,0:3],int_label[:,0:3])))
		X_ul = int_unlabel #int_ulabel
		Y_l = np.float32(np.concatenate((bound[:,3], int_label[:,3])))
		Y_l = np.reshape(Y_l, (len(Y_l),1))

		is_boundary = tf.fill(bound.shape[0], True)
		is_not_boundary = tf.fill(int_label.shape[0] + int_unlabel.shape[0], False)
		is_boundary_all = tf.concat([is_boundary, is_not_boundary], axis=0)

		grad_reg = get_wave_reg(configs.gradient_loss, configs)
		#grad_reg = get_target(configs.target, configs.gradient_loss, configs)

		grad_bools = tf.fill(X_l.shape[0] + X_ul.shape[0], True)

		# if grad_reg == 'unknown':
		# 	grad_bools = tf.fill(X_l.shape[0] + X_ul.shape[0], True)
		# elif grad_reg == "TBD":
		# 	grad_bools = tf.fill(X_l.shape[0] + X_ul.shape[0], True)
		# elif grad_reg == "We'll figure it out":
		# 	grad_bools = tf.fill(X_l.shape[0] + X_ul.shape[0], True)
		# else:
		# 	raise ValueError("Unknown gradient regularizer ", grad_reg)


		error_metrics = {
			"interpolation error (t <= 1)" : lambda model : compute_error_wave(model, int_test),
			"extrapolation error (1 < t)" : lambda model : compute_error_wave(model, ext_test)
		}

		error_plots = {
			"Error vs. time" : lambda model : error_time(model, int_test, ext_test, figs_folder, '/error_time')
		}
		
		print(f"Loaded wave eq. simulation inputs/outputs. Count: {len(X_l)}")

	#Creates labels to pass through network
	is_labeled_l = tf.fill(X_l.shape[0], True)
	is_labeled_ul = tf.fill(X_ul.shape[0], False)

	if X_ul.shape[0] == 0:
		label_bools = is_labeled_l
	else:
		label_bools = tf.concat([is_labeled_l, is_labeled_ul], axis=0)

	Y_ul = tf.zeros((X_ul.shape[0], 1))

	# Add noise to y-values
	if configs.noise > 0:
		Y_l += tf.random.normal(Y_l.shape, stddev=configs.noise)
		Y_l += np.reshape(configs.noise*np.random.randn((len(Y_l))),(len(Y_l),1))

	#Concat inputs, outputs, and bools
	if X_ul.shape[0] == 0:
		X_all = X_l
		Y_all = Y_l
	else:
		X_all = tf.concat([X_l, X_ul], axis=0)
		Y_all = tf.concat([Y_l, Y_ul], axis=0)
	
	

		# if "data-distribution" in configs.plots:
		# 	plot_data(X_l, X_ul, figs_folder, configs)

	print(f"Loaded wave eq. simulation inputs/outputs. Count: {len(X_l)}")
	# else:
	# 	raise ValueError("Unknown data source " + configs.source)

	return X_all, Y_all, label_bools, grad_bools, grad_reg, error_metrics, error_plots

def plot_data(X_l, X_ul, figs_folder, configs):
	
	if X_l.shape[1] == 2:
		# 2D Plotting
		plot_data_2D(X_l, X_ul, figs_folder)
	elif X_l.shape[1] == 3:
		# TODO: 3d plots here
		#plot_data_3D(X_l, X_ul, figs_folder)
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
		for t in tf.range(0, 2 + 1e-3, 0.5):
			class m:
				@staticmethod
				def predict(X):
					ml_input = tf.concat([X, tf.fill((len(X), 1), t)], axis=1)
					return model.predict(ml_input)
			f = lambda x, y: tf.zeros_like(x)
			buf = plot_gridded_functions(m, f, 0, 1, f"_t={t:.3f}", folder=figs_folder)
		# 3D Plotting
		if "heatmap" in configs.plots:
			make_heatmap_movie(model, figs_folder, time_steps = 100, dx = .01, sample_step = .01)
		make_movie(model, figs_folder)
		make_wave_plot(model, t = 0, f_true = 0, figs_folder = figs_folder, tag='0')
		make_wave_plot(model, t = .25, f_true = 0, figs_folder = figs_folder, tag='0.25')
		make_wave_plot(model, t = .5, f_true = 0, figs_folder = figs_folder, tag='0.5')
		make_wave_plot(model, t = .75, f_true = 0, figs_folder = figs_folder, tag='0.75')
		make_wave_plot(model, t = 1, f_true = 0, figs_folder = figs_folder, tag='1')
	
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

	# Memory allocation
	if configs.be_greedy:
		for i in range(5):
			print("I'M GREEDY. SHHHHH DON'T TELL MOM...")
	else:
		physical_devices = tf.config.list_physical_devices('GPU')
		try:
			tf.config.experimental.set_memory_growth(physical_devices[0], True)
		except:
			# Invalid device or cannot modify virtual devices once initialized.
			pass

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

	# ------------------------------------------------------------------------------
	# General setup
	# ------------------------------------------------------------------------------
	# Set seeds for reproducibility
	# np.random.seed(configs.seed)
	# tf.random.set_seed(configs.seed)

	# ------------------------------------------------------------------------------
	# Data preparation
	# ------------------------------------------------------------------------------

	# Get the "interesting" data
	X_all, Y_all, label_bools, grad_bools, grad_reg, error_metrics, error_plots = get_data(configs, figs_folder)#, error_metrics = get_data(configs)

	# Create TensorFlow dataset for passing to 'fit' function (below)
	if configs.from_tensor_slices:
		dataset = tf.data.Dataset.from_tensor_slices((X_all, Y_all, label_bools, grad_bools))
	else:
		dataset = tf.data.Dataset.from_tensors((X_all, Y_all, label_bools, grad_bools))

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
	if configs.from_tensor_slices:
		dataset = dataset.batch(opt_batch_size)
		model.is_dataset_prebatched = True
	else:
		model.is_dataset_prebatched = False
	model.set_gd_noise(configs.gd_noise)

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

	'''
	Y_pred_l = model.predict(X_l)
	loss_value = model.loss_function_f(Y_pred_l, Y_l)/X_l.shape[0]
	print("FINAL Loss: \t\t\t {:.6E}".format(loss_value))
	final_metrics['loss_value'] = "{:.6E}".format(loss_value)
	'''
	for error_name, error_func in error_metrics.items():
		error_val = error_func(model)
		print('FINAL Error/' + error_name + f":\t\t\t {error_val:.6E}")
		final_metrics[error_name] = "{:.6E}".format(error_val)

	if configs.source == 'wave':
		for error_name, error_func in error_plots.items():
			error_val = error_func(model)
		print("Printing error over time")

	if "extrapolation" in configs.plots:
		comparison_plots(model, figs_folder, configs)


	train_time = toc - tic 
	final_metrics['training_time'] = "{:.2F} s".format(train_time)

	os.makedirs(results_dir, exist_ok=True)
	with open(results_dir + '/results.yaml', 'w') as outfile:
		yaml.dump(final_metrics, outfile, default_flow_style=False)