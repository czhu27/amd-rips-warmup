import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import os
import io
import datetime
import argparse
from tensorflow.python.ops.gen_array_ops import size

import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

from helpers import Configs
from nn import create_nn
from targets import get_target

def plot_data(X_f, tag, save_dir):
	plt.scatter(X_f[:,0], X_f[:,1], s=2)
	plt.savefig(save_dir + "/data_" + tag)
	plt.clf()

def data_creation(params, corners):
	N_f_int = params[0]
	N_f_ext = params[1]
	N_f_border = params[2]
	N_f_intl = int(N_f_int * params[3])
	N_f_extl = int(N_f_ext * params[4])
	N_f_intul = int(N_f_int * (1 - params[3]))
	N_f_extul = int(N_f_ext * (1 - params[4]))
	X_f_int = np.zeros((N_f_int,2), dtype = np.float32)
	X_f_ext = np.zeros((N_f_ext,2), dtype = np.float32)
	X_f_border = np.zeros((N_f_border,2), dtype = np.float32)
	#Interior Points on [-1,1]
	X_f_int[:,0] = np.random.uniform(-1, 1, N_f_int)
	X_f_int[:,1] = np.random.uniform(-1, 1, N_f_int)
	#Exterior Points [-2,-1]U[1,2]
	def bad_zone(v):
		if -1 <= v[0] <= 1 and -1 <= v[1] <= 1:
			return True
		else:
			return False

	def generate_point():
		v = np.zeros((2,))
		while bad_zone(v):
			v = np.random.uniform(-2,2,(2,))
		v = v[None,:]
		return v

	for i in range(N_f_ext):
		temp = generate_point()
		X_f_ext[i,:] = temp
		
	#Border Points on box with (|x|,|y|) = (1,1)
	X_f_borderleft = np.array((-1*np.ones((N_f_border//4)), np.random.rand(N_f_border//4))).T
	X_f_borderright = np.array((np.ones((N_f_border//4)), np.random.rand(N_f_border//4))).T
	X_f_borderup = np.array((np.random.rand(N_f_border//4), np.ones((N_f_border//4)))).T
	X_f_borderdown = np.array((np.random.rand(N_f_border//4), -1*np.ones((N_f_border//4)))).T
	X_f_bordermore = np.array((np.random.rand(N_f_border%4), -1*np.ones((N_f_border%4)))).T
	X_f_border[:,0] = np.concatenate((X_f_borderleft[:,0], X_f_borderright[:,0],
                            X_f_borderup[:,0], X_f_borderdown[:,0], X_f_bordermore[:,0]))
	X_f_border[:,1] = np.concatenate((X_f_borderleft[:,1], X_f_borderright[:,1],
                            X_f_borderup[:,1], X_f_borderdown[:,1],X_f_bordermore[:,1]))

	#Labeling Data
	X_f_l = np.zeros((N_f_intl + N_f_border + N_f_extl,2), dtype = np.float32)
	X_f_ul = np.zeros((N_f_intul + N_f_extul,2), dtype = np.float32)
	X_f_l[0:N_f_intl] = X_f_int[0:N_f_intl]
	X_f_l[N_f_intl:N_f_intl + N_f_border] = X_f_border[:,:]
	X_f_l[N_f_intl + N_f_border:] = X_f_ext[0:N_f_extl,:]
	X_f_ul[0:N_f_intul] = X_f_int[N_f_intl:]
	X_f_ul[N_f_intul:] = X_f_ext[N_f_extl:]

	#Add corners
	if corners:
		X_f_l[0, 0] = -2.0; X_f_l[0, 1] = -2.0
		X_f_l[1, 0] =  2.0; X_f_l[1, 1] = -2.0
		X_f_l[2, 0] =  2.0; X_f_l[2, 1] =  2.0
		X_f_l[3, 0] = -2.0; X_f_l[3, 1] =  2.0

	return X_f_l, X_f_ul


'''
Create a meshgrid on the square [lb, ub] with ((ub-lb)/step_size + 1)^2 points
'''
def create_meshgrid(lb, ub, step_size=0.01):
	x0 = np.arange(lb, ub+step_size, step_size)
	x1 = np.arange(lb, ub+step_size, step_size)
	return np.meshgrid(x0, x1), x0.size

'''
Compute L2-error of model against f, on the square [lb, ub]
'''
def compute_error(model, f, lb, ub):
	mesh, n1d = create_meshgrid(lb, ub)
	x0_g, x1_g = mesh
	npts = n1d*n1d

	f_true = f(x0_g, x1_g)

	ml_input = np.zeros((npts, 2))
	ml_input[:,0] = x0_g.flatten()
	ml_input[:,1] = x1_g.flatten()
	ml_output = model.predict(ml_input)
	
	f_ml = np.reshape(ml_output, (n1d, n1d), order = 'C')
	
	error = np.sqrt(np.mean(np.square(f_ml - f_true)))
	return error

'''
i_lb: lower inner bound, i_ub: upper inner bound
o_lb: lower outer bound, o_ub: upper outer bound
Compute L2-error of model against f, on the square [o_lb, o_ub], excluding the
points in the square [i_lb, i_ub]
'''
def extrap_error(model, f, i_lb, i_ub, o_lb, o_ub, step_size=0.01):
	mesh, n1d = create_meshgrid(o_lb, o_ub, step_size)
	x, y = mesh
	npts = n1d*n1d
	less_points = int((i_ub-i_lb)/step_size)+1
	npts = npts - less_points**2
	
	is_interior = ((x >= i_lb) & (x <= i_ub+step_size)) & ((y >= i_lb) & (y <= i_ub+step_size))
	x_ext = x[~is_interior]
	y_ext = y[~is_interior]

	f_true = f(x_ext, y_ext)

	ml_input = np.zeros((npts, 2))
	ml_input[:,0] = x_ext.flatten()
	ml_input[:,1] = y_ext.flatten()
	ml_output = model.predict(ml_input)

	f_ml = np.reshape(ml_output, (npts), order = 'C')

	error = np.sqrt(np.mean(np.square(f_ml - f_true)))
	return error

def plot_gridded_functions(model, f, lb, ub, tag, folder="figs"):
	n1d = 101
	npts = n1d*n1d
	x0 = np.linspace(lb, ub, n1d)
	x1 = np.linspace(lb, ub, n1d)
	x0_g, x1_g = np.meshgrid(x0, x1)

	# Compute true function values
	f_true = f(x0_g, x1_g)

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


def main(configs: Configs):
	# Setup folder structure vars
	run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	output_dir = configs.output_dir + "/" + run_name
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
	np.random.seed(0)
	tf.random.set_seed(0)

	# ------------------------------------------------------------------------------
	# Data preparation
	# ------------------------------------------------------------------------------
	# Data for training NN based on L_f loss function
	X_f_l, X_f_ul = data_creation(configs.dataset, configs.corners)

	# Set target function
	f, grad_reg = get_target(configs.target, configs.gradient_loss, configs)
	# f = lambda x,y : parabola(x,y, configs.f_a, configs.f_b)

	f_true = f(X_f_l[:, 0:1], X_f_l[:, 1:2])
	f_ul = tf.zeros((X_f_ul.shape[0], 1))

	if configs.noise > 0:
		f_true += np.reshape(configs.noise*np.random.randn((len(f_true))),(len(f_true),1))
		f_ul += np.reshape(configs.noise*np.random.randn((len(f_ul))),(len(f_ul),1))

	is_labeled_l = tf.fill(f_true.shape, True)
	is_labeled_ul = tf.fill(f_ul.shape, False)

	X_f_all = tf.concat([X_f_l, X_f_ul], axis=0)
	f_all = tf.concat([f_true, f_ul], axis=0)
	is_labeled_all = tf.concat([is_labeled_l, is_labeled_ul], axis=0)

	if "data-distribution" in configs.plots:
		print("Saving data distribution plots")
		plot_data(X_f_l, "labeled", figs_folder)
		plot_data(X_f_ul, "unlabeled", figs_folder)
		plot_data(X_f_all, "all", figs_folder)
	

	# Create TensorFlow dataset for passing to 'fit' function (below)
	dataset = tf.data.Dataset.from_tensors((X_f_all, f_all, is_labeled_all))

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
	f_pred_0 = model.predict(X_f_l)
	error_0 = np.sqrt(np.mean(np.square(f_pred_0 - f_true)))

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
	f_pred_1 = model.predict(X_f_l)
	error_1 = np.sqrt(np.mean(np.square(f_pred_1 - f_true)))
	loss_value = model.loss_function_f(f_pred_1, f_true)/X_f_l.shape[0]

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

def make_configs(changes_configs):
	# Load dict from yaml file
	default_configs = yaml.safe_load(open('configs/default.yaml'))

	# Merge the two configs
	configs = default_configs.copy()
	configs.update(changes_configs)

	# Convert dict to object
	configs = Configs(**configs)
	
	return configs

def get_filename(path):
	return os.path.basename(path).split(".")[0]

def grid_search(search_file):
	print("Running a grid search.")
	print("YAML file: ", search_file)
	search_configs = yaml.safe_load(open(search_file))
	assert len(search_configs) == 1, "Only supports grid search in one argument"
	key = list(search_configs.keys())[0]
	values = search_configs[key]
	all_configs = []
	for i, value in enumerate(values):
		changes_configs = {key: value}
		configs = make_configs(changes_configs)

		search_file_name = get_filename(search_file)
		if isinstance(value, (list, tuple)):
			value_name = '_'.join(str(i) for i in value)
		else:
			value_name = str(value)
		configs.output_dir = "output/search/" + search_file_name + f"/{key}={value_name}"

		all_configs.append(configs)
		
	for configs in all_configs:
		main(configs)

def single_run(changes_file):
	print("Running a single run.")
	print("YAML file: ", changes_file)
	changes_configs = yaml.safe_load(open(changes_file))
	configs = make_configs(changes_configs)

	changes_file_name = get_filename(changes_file)
	configs.output_dir = "output/single/" + changes_file_name

	main(configs)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-g", "--grid-search", type=str, default=None)
	parser.add_argument("-s", "--single-run", type=str, default="configs/single/test.yaml")
	args = parser.parse_args()

	
	if args.grid_search is not None:
		grid_search(args.grid_search)
	
	else:
		single_run(args.single_run)