# ------------------------------------------------------------------------------
# Contains custom neural network model and associated functions
# ------------------------------------------------------------------------------
import numpy as np
import math

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.python.ops.gen_math_ops import lgamma
from tensorflow.python.types.core import Value

def nth_gradient(y, x, n, tape):
	'''
	Compute the nth order gradient of y wrt x (using tape)
	'''
	grad = y
	for i in range(n):
		grad = tape.gradient(grad, x)
	return grad

def gradient_condition(f, x, y, tape):
	fxx = nth_gradient(f, x, 2, tape)
	fyy = nth_gradient(f, y, 2, tape)
	fxxy = tape.gradient(fxx, y)
	fyyx = tape.gradient(fyy, x)
	fxxx = tape.gradient(fxx, x)
	fyyy = tape.gradient(fyy, y)

	grads = tf.concat([fxxx, fxxy, fyyx, fyyy], axis=0)
	grad_loss = tf.math.reduce_sum(tf.math.abs(grads))
	return grad_loss

# ------------------------------------------------------------------------------
# Custom model based on Keras Model.
# ------------------------------------------------------------------------------

# TODO: Make attribute of NN
loss_tracker = keras.metrics.Mean(name="loss")

class NN(keras.models.Model):

	# Custom loss for function value
	def loss_function_f(self, f, f_pred):
		sq_diff = tf.math.squared_difference(f, f_pred)
		loss_value = tf.math.reduce_sum(sq_diff)
		return loss_value			 
	# def loss_function_f
	
	# Called from outside as a convenient way (albeit slightly risky)
	# to specify the mini-batch size
	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
	
	# Redefine train_step used for optimizing the neural network parameters
	# This function implements one epoch (one pass over entire dataset)
	def train_step(self, batch):
		
		# Retrieve size of entire dataset
		X_f, f, is_labeled = batch
		batch_size = self.batch_size
			
		# For gradient of loss w.r.t. trainable variables	
		with tf.GradientTape(persistent=True) as tape:

			if self.gradient_loss:
				# Allow gradients wrt X_f
				tape.watch(X_f)
			
			# Forward run
			# Allow for gradients w.r.t. individual columns (x,y)
			xy = tf.unstack(X_f, axis=1)
			x,y = xy
			new_X_f = tf.stack(xy, axis=1)
			X_f = new_X_f
			f_pred = self.call(X_f)

			L_f = 0
			# Compute L_f: \sum_i |f_i - f_i*|^2
			L_f += self.loss_function_f(f[is_labeled], f_pred[is_labeled])

			if self.gradient_loss:
				grad_loss = gradient_condition(f_pred, x, y, tape)
				#grad_loss_ext = gradient_condition(f_pred_ext, x_ext, y_ext, tape)
				
				condition_weight = 1
				L_f += condition_weight * grad_loss #+ grad_loss_ext
					
			# Add regularization loss	
			L_f += sum(self.losses)
		
		
		# Compute gradient of total loss w.r.t. trainable variables
		trainable_vars = self.trainable_variables
		# abs_list = [tf.math.abs(w) for w in trainable_vars]
		# sum_list = [tf.reduce_sum(x) for x in abs_list]
		# w_l1 = sum(sum_list)
		# L_f += w_l1
		gradients = tape.gradient(L_f, trainable_vars)
	
		# Update network parameters
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
	
		# Compute our own metrics	
		loss_value = L_f / batch_size
		loss_tracker.update_state(loss_value)

		# Update loss and return value
		# print(loss_value)
		return {"loss": loss_tracker.result()}
		
	# def train_step

	@property
	def metrics(self):
		# We list our `Metric` objects here so that `reset_states()` can be
		# called automatically at the start of each epoch
		# or at the start of `evaluate()`.
		# If you don't implement this property, you have to call
		# `reset_states()` yourself at the time of your choosing.
		return [loss_tracker]

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

def create_nn(layer_widths, configs):
	num_hidden_layers = len(layer_widths) - 2
	
	# Weight initializer
	initializer = keras.initializers.GlorotUniform()
	
	# Create input layer
	input_layer = keras.Input(layer_widths[0], 
							  name = 'input')

	# Create hidden layers
	layer = input_layer
	for i in range(num_hidden_layers):
		width = layer_widths[i + 1]
		name = 'h' + str(i)
		layer = keras.layers.Dense(width, 
							 	   activation=configs.activation, name=name,
							 	   kernel_initializer=initializer,
									kernel_regularizer=get_regularizer(configs),
									#kernel_regularizer=tf.keras.regularizers.L1L2(l1=lam, l2=lam),
									)(layer)

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
	model.gradient_loss = configs.gradient_loss
	return model

# def create_nn	
