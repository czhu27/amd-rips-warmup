import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import os
import io
import yaml


import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import losses
from tensorflow.keras import optimizers

from nn import NN, create_nn

import argparse

import datetime

def main(configs):
	#tf.enable_eager_execution()

	parser = argparse.ArgumentParser()
	parser.add_argument("--exp-id", default = "no_exp_id")
	args = parser.parse_args()

	class args:
		exp_id = "no_exp_id"
	exp_folder = "output/" + args.exp_id + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	log_dir = exp_folder + "/logs" 
	scalar_dir = log_dir + "/scalars"
	file_writer = tf.summary.create_file_writer(scalar_dir + "/metrics")
	file_writer.set_as_default()

	def compute_error(lb, ub):
		n1d = 101
		npts = n1d*n1d
		x0 = np.linspace(lb, ub, n1d)
		x1 = np.linspace(lb, ub, n1d)
		x0_g, x1_g = np.meshgrid(x0, x1)
		
		f_true = f_a*np.square(x0_g) + f_b*np.square(x1_g)

		ml_input = np.zeros((npts, 2))
		ml_input[:,0] = x0_g.flatten()
		ml_input[:,1] = x1_g.flatten()
		ml_output = model.predict(ml_input)
		
		f_ml = np.reshape(ml_output, (n1d, n1d), order = 'C')
		
		error = np.sqrt(np.mean(np.square(f_ml - f_true)))
		return error;

	# ------------------------------------------------------------------------------
	# General setup
	# ------------------------------------------------------------------------------
	# Set seeds for reproducibility
	np.random.seed(0)
	tf.random.set_seed(0)

	# Function coefficients (f = f_a*x^2 + f_b*y^2)
	f_a = 1.0
	f_b = 1.0

	# ------------------------------------------------------------------------------
	# Data preparation
	# ------------------------------------------------------------------------------
	# Data for training NN based on L_f loss function
	N_f = configs.num_data
	X_f = np.zeros((N_f, 2), dtype = np.float32)
	X_f[:, 0] = 2*np.random.rand(N_f) - 1
	X_f[:, 1] = 2*np.random.rand(N_f) - 1
	#X_f[0, 0] = -2.0; X_f[0, 1] = -2.0
	#X_f[1, 0] =  2.0; X_f[1, 1] = -2.0
	#X_f[2, 0] =  2.0; X_f[2, 1] =  2.0
	#X_f[3, 0] = -2.0; X_f[3, 1] =  2.0
	f = f_a*X_f[:, 0:1] ** 2 + f_b*X_f[:, 1:2] ** 2	# f_a*x^2 + f_b*y^2

	# Create TensorFlow dataset for passing to 'fit' function (below)
	dataset = tf.data.Dataset.from_tensors((X_f, f))

	# ------------------------------------------------------------------------------
	# Create neural network (physics-inspired)
	# ------------------------------------------------------------------------------
	layers = configs.layers
	model = create_nn(layers, configs)
	model.summary()

	# ------------------------------------------------------------------------------
	# Assess accuracy with non-optimized model
	# ------------------------------------------------------------------------------
	f_pred_0 = model.predict(X_f)
	error_0 = np.sqrt(np.mean(np.square(f_pred_0 - f)))

	# ------------------------------------------------------------------------------
	# Model compilation / training (optimization)
	# ------------------------------------------------------------------------------
	opt_step = configs.lr			# gradient descent step
	opt_batch_size = configs.batch_size	# batch size
	opt_num_its = configs.epochs		# number of iterations

	model.set_batch_size(opt_batch_size)

	optimizer = optimizers.Adam(learning_rate = opt_step)
	model.compile(optimizer = optimizer,) #run_eagerly=True)		# DEBUG
	tic = time.time()

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
				error1 = compute_error(-1.0, 1.0)
				tf.summary.scalar('Error/interpolation', data=error1, step=epoch)
				error2 = compute_error(-2.0, 2.0)
				tf.summary.scalar('Error/extrapolation', data=error2, step=epoch)

	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	logging_callbacks = [TimeLogger(), StressTestLogger(), tensorboard_callback]

	if configs.plots:
		callbacks = logging_callbacks
	else:
		callbacks = []

	model.fit(dataset, 
			epochs=opt_num_its, 
			verbose=2,
			callbacks=callbacks)
	toc = time.time()
	print("Training time: {:.2F} s\n".format(toc - tic))
	#model.save("model.h5")

	# ------------------------------------------------------------------------------
	# Assess accuracy with optimized model and compare with non-optimized model
	# ------------------------------------------------------------------------------
	f_pred_1 = model.predict(X_f)
	error_1 = np.sqrt(np.mean(np.square(f_pred_1 - f)))

	print("Train set error (before opt): {:.15E}".format(error_0))
	print("Train set error (after opt) : {:.15E}".format(error_1))
	print("Ratio of errors             : {:.1F}".format(error_0/error_1))

	# ------------------------------------------------------------------------------
	# Stress set - Assess extrapolation capabilities
	# ------------------------------------------------------------------------------


	def plot_gridded_functions(lb, ub, tag, folder="figs"):

		n1d = 101
		npts = n1d*n1d
		x0 = np.linspace(lb, ub, n1d)
		x1 = np.linspace(lb, ub, n1d)
		x0_g, x1_g = np.meshgrid(x0, x1)

		# Compute true function values
		f_true = f_a*np.square(x0_g) + f_b*np.square(x1_g)

		# Compute ML function values
		ml_input = np.zeros((npts, 2))
		ml_input[:,0] = x0_g.flatten()
		ml_input[:,1] = x1_g.flatten()
		ml_output = model.predict(ml_input)
		f_ml = np.reshape(ml_output, (n1d, n1d), order = 'C')

		fig = plt.figure()
		fig.set_figheight(8)
		fig.set_figwidth(8)
		fig.tight_layout()
		ax = fig.add_subplot(221, projection='3d')
		ax.plot_surface(x0_g, x1_g, f_true, cmap=cm.coolwarm)
		ax.set_title('True')
		#plt.savefig('figs/true' + str(tag) + '.png')

		ax = fig.add_subplot(222, projection='3d')
		ax.plot_surface(x0_g, x1_g, f_ml, cmap=cm.coolwarm)
		ax.set_title('ML')
		#plt.savefig('figs/ml' + str(tag) + '.png')

		ax = fig.add_subplot(223, projection='3d')
		ax.plot_surface(x0_g, x1_g, np.abs(f_ml - f_true), cmap=cm.coolwarm)
		ax.set_title('|True - ML|')
		#plt.savefig('figs/diff' + str(tag) + '.png')
		plt.savefig(folder + '/all' + str(tag) + '.png')
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		return buf

	# Make grid to display true function and predicted
	error1 = compute_error(-1.0, 1.0)

	def add_image(plot_buf):
		# Convert PNG buffer to TF image
		image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
		# Add the batch dimension
		image = tf.expand_dims(image, 0)
		# Add image summary
		summary_op = tf.summary.image("plot", image, step=0)

	# def plot_to_image(figure):
	# 	"""Converts the matplotlib plot specified by 'figure' to a PNG image and
	# 	returns it. The supplied figure is closed and inaccessible after this call."""
	# 	# Save the plot to a PNG in memory.
	# 	buf = io.BytesIO()
	# 	plt.savefig(buf, format='png')
	# 	# Closing the figure prevents it from being displayed directly inside
	# 	# the notebook.
	# 	plt.close(figure)
	# 	buf.seek(0)
	# 	# Convert PNG buffer to TF image
	# 	image = tf.image.decode_png(buf.getvalue(), channels=4)
	# 	# Add the batch dimension
	# 	image = tf.expand_dims(image, 0)
	# 	return image

	# log_file = exp_folder + "/test.log"
	# logging.basicConfig(filename=log_file,
	# 					filemode='a',
	# 					format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
	# 					datefmt='%H:%M:%S',
	# 					level=logging.DEBUG)
	figs_folder = exp_folder + "/figs"
	os.makedirs(figs_folder, exist_ok=True)
	buf = plot_gridded_functions(-1.0, 1.0, "100", folder=figs_folder);
	#add_image(buf)
	print("Error [-1,1]x[-1,1]: {:.6E}".format(error1))
	error2 = compute_error(-2.0, 2.0)
	buf = plot_gridded_functions(-2.0, 2.0, "200", folder=figs_folder);
	#add_image
	print("Error [-2,2]x[-2,2]: {:.6E}".format(error2))

class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)

if __name__ == "__main__":
	configs = yaml.safe_load(open('configs/default.yaml'))

	configs = Struct(**configs)

	main(configs)