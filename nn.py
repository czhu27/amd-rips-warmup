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

# ------------------------------------------------------------------------------
# Custom model based on Keras Model.
# ------------------------------------------------------------------------------
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

	# Create mini batches
	def create_mini_batches(self, dataset):
		# Batch size should by user using the 'set_batch_size' function
		batch_size = self.batch_size
		
		X_f, f = dataset
		
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
			X_f_b = tf.gather(X_f, indices, axis = 0)
			f_b = tf.gather(f, indices, axis = 0)
			
			batch = (X_f_b, f_b)
			batches.append(batch)
		
		if (num_batches*bs_f < m_f) or (num_batches*bs_df < m_df):
			idx0 = bs_f*num_batches
			idx1 = m_f
			indices = perm_idx_f[idx0:idx1]
			X_f_b = tf.gather(X_f, indices, axis = 0)
			f_b = tf.gather(f, indices, axis = 0)

			batch = (X_f_b, f_b)
			batches.append(batch)
		
		return batches
	# def create_mini_batches	
	
	# Redefine train_step used for optimizing the neural network parameters
	# This function implements one epoch (one pass over entire dataset)
	def train_step(self, dataset):
		
		# Retrieve size of entire dataset
		X_f = dataset[0]
		m_f = X_f.shape[0]
		
		mini_batches = self.create_mini_batches(dataset)	
		
		# Keep track of total loss value for this epoch
		loss_value_f = 0.0
		
		# Loop over all mini-batches
		for batch in mini_batches:
			X_f, f = batch
			
			# For gradient of loss w.r.t. trainable variables	
			with tf.GradientTape() as tape:
				
				# Forward run
				f_pred = self.call(X_f)

				# Compute L_f: \sum_i |f_i - f_i*|^2	
				L_f = self.loss_function_f(f, f_pred)
				
				if self.gradient_loss:
					f_predx = tf.gradients(f_pred, X_f)[0]
					f_predxx = tf.gradients(f_predx, X_f)[0]
					f_predxxx = tf.gradients(f_predxx, X_f)[0]
					grad = tf.math.reduce_sum(tf.math.abs(f_predxxx))
					L_f += sum(self.losses)
				#L_f += grad

				tf.gradients(f_pred, X_f)
				# f_pred_x = tf.gradients(f_pred, X_f)[0]
				# f_pred_xx = tf.gradients(f_pred_x, X_f)[0]
				# f_pred_xxx = tf.gradients(f_pred_xx, X_f)[0]
				# L_f = L_f + f_pred_xxx
			
			
			# Compute gradient of total loss w.r.t. trainable variables
			trainable_vars = self.trainable_variables
			# abs_list = [tf.math.abs(w) for w in trainable_vars]
			# sum_list = [tf.reduce_sum(x) for x in abs_list]
			# w_l1 = sum(sum_list)
			# L_f += w_l1
			gradients = tape.gradient(L_f, trainable_vars)
		
			# Update network parameters
			self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		
			# Increment total loss value by mini-batch-wise contribution
			loss_value_f += L_f

		# end for loop on mini batches
		
		loss_value = loss_value_f/m_f

		# Update loss and return value
		return {"loss": loss_value}
		
	# def train_step

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
