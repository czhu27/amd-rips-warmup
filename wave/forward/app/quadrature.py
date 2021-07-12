import numpy as np
import sys

# ------------------------------------------------------------------------------
# class Gauss2d
# ------------------------------------------------------------------------------
class Gauss2d:
	# Constructor (number of quadrature points in 1D)
	def __init__(self, num_points_1d):
		self.num_points = num_points_1d*num_points_1d
		self.coordinates = np.zeros((self.num_points, 2), dtype = np.float32)
		self.weights = np.zeros(self.num_points, dtype = np.float32)

		# Store quadrature points and weights
		if (num_points_1d == 1):
			# ----------
			xi = np.array([0], dtype = np.float32)
			w = np.array([2.0], dtype = np.float32)
			# ----------
		elif (num_points_1d == 2):
			# ----------
			xi = np.array([-1.0/np.sqrt(3.0), 
							1.0/np.sqrt(3.0)], dtype = np.float32)
			w = np.array([1.0, 
						  1.0], dtype = np.float32)
			# ----------
		elif (num_points_1d == 3):
			# ----------
			xi = np.array([-np.sqrt(0.6),
						    0.0,
						    np.sqrt(0.6)], dtype = np.float32)
			w = np.array([5./9.,
						  8./9.,
						  5./9.], dtype = np.float32)			   
			# ----------
		else:
			error_msg = "ERROR: element not available for p = " + str(self.p)
			sys.exit(error_msg)
		index = 0
		for j in range(len(w)):
			for i in range(len(w)):
				self.coordinates[index][0] = xi[i]
				self.coordinates[index][1] = xi[j]
				self.weights[index] = w[j]*w[i]
				index += 1
		
		# Save 1D weight for 1D integration
		self.weights_1d = w
	# def __init__

# class Gauss2d

# ------------------------------------------------------------------------------
# class GLL2d
# ------------------------------------------------------------------------------
class GLL2d:
	# Constructor (number of quadrature points in 1D)
	def __init__(self, num_points_1d):
		self.num_points = num_points_1d*num_points_1d
		self.coordinates = np.zeros((self.num_points, 2), dtype = np.float32)
		self.weights = np.zeros(self.num_points, dtype = np.float32)

		# Store quadrature points and weights
		if (num_points_1d == 1):
			# ----------
			xi = np.array([0], dtype = np.float32)
			w = np.array([2.0], dtype = np.float32)
			# ----------
		elif (num_points_1d == 2):
			# ----------
			xi = np.array([-1.0, 
							1.0], dtype = np.float32)
			w = np.array([1.0, 
						  1.0], dtype = np.float32)
			# ----------
		elif (num_points_1d == 3):
			# ----------
			xi = np.array([-1.0,
						    0.0,
						    1.0], dtype = np.float32)
			w = np.array([1./3.,
						  4./3.,
						  1./3.], dtype = np.float32)			   
			# ----------
		else:
			error_msg = "ERROR: element not available for p = " + str(self.p)
			sys.exit(error_msg)
		index = 0
		for j in range(len(w)):
			for i in range(len(w)):
				self.coordinates[index][0] = xi[i]
				self.coordinates[index][1] = xi[j]
				self.weights[index] = w[j]*w[i]
				index += 1
		
		# Save 1D weight for 1D integration
		self.weights_1d = w
	# def __init__

# class GLL2d
