# ------------------------------------------------------------------------------
# Contains custom neural network model and associated functions
# ------------------------------------------------------------------------------
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.python.ops.gen_math_ops import lgamma
from tensorflow.python.types.core import Value
from helpers import stack_unstack

# ------------------------------------------------------------------------------
# Custom model based on Keras Model.
# ------------------------------------------------------------------------------
class NN(keras.models.Model):

	# Custom loss for function value
	def loss_function_f(self, f, f_pred):
		sq_diff = tf.math.squared_difference(f, f_pred)
		loss_val = tf.math.reduce_mean(sq_diff)
		loss_value = tf.where(tf.math.is_nan(loss_val), tf.zeros_like(loss_val), loss_val)
		return loss_value			 
	# def loss_function_f
	
	# Called from outside as a convenient way (albeit slightly risky)
	# to specify the mini-batch size
	# def set_batch_size(self, batch_size):
	# 	self.batch_size = batch_size

	# def set_gd_noise(self, gd_noise):
	# 	self.gd_noise = gd_noise

	def get_batch(self, dataset, indices):
		batch = []
		for element in dataset:
			if isinstance(element, dict):
				element_b = {k: tf.gather(v, indices, axis = 0) for k,v in element.items()}
			else:
				element_b = tf.gather(element, indices, axis = 0)
			batch.append(element_b)
		batch = tuple(batch)
		return batch

	# Create mini batches
	def create_mini_batches(self, dataset):
		# Batch size should by user using the 'set_batch_size' function
		batch_size = self.batch_size
		
		# X_f, f, l_bools, g_bools = dataset
		X_f = dataset[0]

		# Number of batches
		m_f = X_f.shape[0]
		num_batches = math.floor(m_f/batch_size)

		# Mini-batch sizes
		bs_f = math.floor(m_f/num_batches)
		
		# Create mini-batches based on random selection
		perm_idx_f = np.random.permutation(m_f)
		batches = []

		for i in range(num_batches):
			idx0 = bs_f*i
			idx1 = bs_f*(i + 1)
			indices = perm_idx_f[idx0:idx1]

			# Batch along an arbitrary number of tensors
			batch = self.get_batch(dataset, indices)

			# X_f_b = tf.gather(X_f, indices, axis = 0)
			# f_b = tf.gather(f, indices, axis = 0)
			# l_bools_b = tf.gather(l_bools, indices, axis = 0)
			
			# batch = (X_f_b, f_b, l_bools_b)
			batches.append(batch)
		
		if (num_batches*bs_f < m_f): #or (num_batches*bs_df < m_df):
			idx0 = bs_f*num_batches
			idx1 = m_f
			indices = perm_idx_f[idx0:idx1]
			
			# Batch along an arbitrary number of tensors
			batch = self.get_batch(dataset, indices)

			# X_f_b = tf.gather(X_f, indices, axis = 0)
			# f_b = tf.gather(f, indices, axis = 0)
			# l_bools_b = tf.gather(l_bools, indices, axis = 0)

			#batch = (X_f_b, f_b, l_bools_b)
			batches.append(batch)
		
		return batches
	# def create_mini_batches
	
	def do_one_batch(self, batch):
		X_f, f, l_bools, grad_bools = batch

		# f = tf.reshape(f, [f.shape[0],1])
		
		# For gradient of loss w.r.t. trainable variables	
		with tf.GradientTape(persistent=True) as tape:

			# Allow gradients wrt X_f
			tape.watch(X_f)
			# TODO: Move source to back ;)
			# TODO: Dont calc grad w r t source x y
			new_X_f, xyz = stack_unstack(X_f)
			X_f, old_X_f = new_X_f, X_f
	
			# Calc. model predicted y values
			f_pred = self.call(X_f)

			## Compute Loss
			# Compute L_f: \sum_i |f_i - f_i*|^2
			L_f = 0
			base_loss = self.loss_function_f(f[l_bools], f_pred[l_bools])
			self.base_loss_tracker.update_state(base_loss)
			# Hacky silent add
			w_base_loss = self.base_condition_weight.value() * base_loss
			self.w_base_loss_tracker.update_state(w_base_loss)
			L_f += w_base_loss

			if self.gradient_loss:
				# Compute gradient condition (deviation from diff. eq.)
				grad_loss = 0
				for region, gr_list in self.grad_regs.items():
					region_mask = grad_bools[region]
					# X_f, xyz = stack_unstack(old_X_f[idx])
					# f_pred = self.call(X_f)
					for grad_reg in gr_list:
						v = grad_reg(f_pred, xyz[-4:-1], tape)
						# TODO: Do we need this?
						v = tf.cast(v, tf.float32)
						v_masked = v[region_mask]
						# assert v_masked.shape[0] == tf.reduce_sum(tf.cast(region_mask, tf.float32))
						# L1 norm
						grad_loss += tf.math.reduce_mean(tf.math.abs(v))

				# grad_loss = self.gradient_regularizer(f_pred, xyz, tape)
				self.grad_reg_loss_tracker.update_state(grad_loss)
				w_grad_loss = self.grad_condition_weight.value() * grad_loss
				self.w_grad_reg_loss_tracker.update_state(w_grad_loss)
				L_f += w_grad_loss

			# Add regularization loss	
			weighted_reg_loss = sum(self.losses)
			self.w_reg_loss_tracker.update_state(weighted_reg_loss)
			L_f += weighted_reg_loss 
		
		
		# Compute gradient of total loss w.r.t. trainable variables
		trainable_vars = self.trainable_variables
		# abs_list = [tf.math.abs(w) for w in trainable_vars]
		# sum_list = [tf.reduce_sum(x) for x in abs_list]
		# w_l1 = sum(sum_list)
		# L_f += w_l1
		gradients = tape.gradient(L_f, trainable_vars)

		# Add noise
		if self.gd_noise > 0:
			noisy_gradients = [g + tf.random.normal(g.shape, stddev=self.gd_noise) for g in gradients]
			gradients = noisy_gradients
	
		# Update network parameters
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update loss metric
		self.loss_tracker.update_state(L_f)

	# Redefine train_step used for optimizing the neural network parameters
	# This function implements one epoch (one pass over entire dataset)
	def train_step(self, dataset):
		
		if self.is_dataset_prebatched:
			batch = dataset
			self.do_one_batch(batch)
		
		else:
			mini_batches = self.create_mini_batches(dataset)	
		
			# Loop over all mini-batches
			for batch in mini_batches:
				self.do_one_batch(batch)
			# end for loop on mini batches

		# Update loss and return value
		return {m.name: m.result() for m in self.metrics}
		
	# def train_step

	@property
	def metrics(self):
		# We list our `Metric` objects here so that `reset_states()` can be
		# called automatically at the start of each epoch
		# or at the start of `evaluate()`.
		# If you don't implement this property, you have to call
		# `reset_states()` yourself at the time of your choosing.

		# return [self.loss_tracker, self.base_loss_tracker, 
		# 	self.grad_reg_loss_tracker, self.w_reg_loss_tracker]
		return [self.loss_tracker]

# class NN

# ------------------------------------------------------------------------------
# Create neural network
# layers: array of layer widths, including input (0) and output (last)
# activation: activation function (string: 'tanh', 'relu', etc.)
# ------------------------------------------------------------------------------
def get_regularizer(configs):
	if configs.regularizer == 'none':
		return None
	elif configs.regularizer == "l1":
		return tf.keras.regularizers.L1(l1=configs.reg_const)
	elif configs.regularizer == "l2":
		return tf.keras.regularizers.L2(l2=configs.reg_const)
	elif configs.regularizer == "l1l2":
		return tf.keras.regularizers.L1L2(l1=configs.reg_const, l2=configs.reg_const)
	else:
		raise ValueError("Unknown regularizer")

def relu_squared(x):
	x = (K.relu(x))**2
	return x

def create_nn(layer_widths, configs):
	num_hidden_layers = len(layer_widths) - 2
	
	# Weight initializer
	initializer = keras.initializers.GlorotUniform()
	
	# Create input layer
	input_layer = keras.Input(layer_widths[0], 
							  name = 'input')

	# Process dropout
	if isinstance(configs.dropout_rates, list) or isinstance(configs.dropout_rates, tuple):
		assert len(configs.dropout_rates) == num_hidden_layers, "Wrong number of dropout rates."
		dropout_rates = configs.dropout_rates
	elif isinstance(configs.dropout_rates, float):
		dropout_rates = num_hidden_layers * [configs.dropout_rates]
	else:
		raise ValueError("Invalid dropout_rates in config")

	# Create hidden layers
	layer = input_layer
	for i in range(num_hidden_layers):
		print("Layer " + str(i))
		width = layer_widths[i + 1]
		name = 'h' + str(i)
		layer = keras.layers.Dense(width,
							 	   activation=configs.activation, name=name,
							 	   kernel_initializer=initializer,
									kernel_regularizer=get_regularizer(configs),
									#kernel_regularizer=tf.keras.regularizers.L1L2(l1=lam, l2=lam),
									)(layer)
		layer = keras.layers.Dropout(dropout_rates[i])(layer)

	# Create output layer
	width = layer_widths[len(layer_widths) - 1]
	output_layer = keras.layers.Dense(width, 
									  name = 'output',
									  kernel_initializer = initializer,
									  kernel_regularizer=get_regularizer(configs),
									  #kernel_regularizer=tf.keras.regularizers.L1L2(l1=lam, l2=lam),
									  )(layer)

	

	# Model
	model = NN(inputs=input_layer, outputs=output_layer)
	if configs.gradient_loss == "none":
		model.gradient_loss = False
	else:
		model.gradient_loss = True

	model.grad_condition_weight = tf.Variable(configs.grad_reg_const, dtype = tf.float32, trainable = False)
	model.base_condition_weight = tf.Variable(1, dtype = tf.float32, trainable = False)
	
	model.loss_tracker = tf.keras.metrics.Mean(name='loss')
	model.base_loss_tracker = tf.keras.metrics.Mean(name='base_loss')
	model.w_base_loss_tracker = tf.keras.metrics.Mean(name='w_base_loss')
	model.grad_reg_loss_tracker = tf.keras.metrics.Mean(name='grad_reg_loss')
	model.w_grad_reg_loss_tracker = tf.keras.metrics.Mean(name='w_grad_reg_loss')
	model.w_reg_loss_tracker = tf.keras.metrics.Mean(name='w_reg_loss')
	
	return model

# def create_nn	
