import numpy as np
import sys

# ------------------------------------------------------------------------------
# class Mesh
# ------------------------------------------------------------------------------
class Element2d:
	# Constructor (p = polynomial degree)
	def __init__(self, p):
		self.p = p
		self.num_nodes = (p + 1)*(p + 1)
		self.__build()
	# def __init__

	# Calculate element info
	def __build(self):
		# Define local coordinates in 1D (local coordinates
		# in 2D directly derive from the 1D coordinates)
		if (self.p == 0):
			self.xi = np.array([0.0], 
							   dtype = np.float32)
		elif (self.p == 1):
			self.xi = np.array([-1.0, 1.0], 
							   dtype = np.float32)
		elif (self.p == 2):
			self.xi = np.array([-1.0, 0.0, 1.0], 
							   dtype = np.float32)
		elif (self.p == 3):
			self.xi = np.array([-1.0, -np.sqrt(.2), np.sqrt(.2), 1.0], 
							   dtype = np.float32)
		else:
			error_msg = "ERROR: element not available for p = " + str(self.p)
			sys.exit(error_msg)
		# if	

		# Define 2D coordinates (correspond to nodes)
		n1d = len(self.xi)
		n = n1d*n1d
		self.coordinates = np.zeros((n, 2), dtype = np.float32)
		index = 0
		for j in range(n1d):
			for i in range(n1d):
				self.coordinates[index, 0] = self.xi[i]
				self.coordinates[index, 1] = self.xi[j]
				index += 1
	# def __build
	
	# Evaluate 2D shape functions at local coordinates (xi_i, eta_i)
	# local_coordinates: array is of shape (num_points, 2)
	# Return: array psi of shape functions evaluated at (xi_i, eta_i).  
	#         Array is of shape(num_nodes, num_points).
	def evaluate_psi(self, local_coordinates):
		num_points = local_coordinates.shape[0]
		num_nodes = (self.p + 1)**2
		psi = np.zeros((num_nodes, num_points), dtype = np.float32)
		
		if (self.p == 1):
			psi_1d_xi  = self.__psi_p1(local_coordinates[:,0])
			psi_1d_eta = self.__psi_p1(local_coordinates[:,1])
		elif (self.p == 2):
			psi_1d_xi  = self.__psi_p2(local_coordinates[:,0])
			psi_1d_eta = self.__psi_p2(local_coordinates[:,1])
		else:
			error_msg = "ERROR: element not available for p = " + str(self.p)
			sys.exit(error_msg)
		# if	
		
		# Calculate 2D shape functions as products of 1D shape functions
		# TODO: vectorize (outer product and flatten)
		for j in range(num_points):
			i = 0;
			for i_eta in range(self.p + 1):
				for i_xi in range(self.p + 1):
					psi[i,j] = psi_1d_eta[i_eta,j]*psi_1d_xi[i_xi,j]
					i += 1
				# for i_xi
			# for i_eta
		# for j

		return psi
	# def evaluate_psi

	# Evaluate 2D shape function derivatives at local coordinates
	# local_coordinates: array is of shape (num_points, 2)
	# Return: dpsi0 (dpsi/dxi) and dpsi1 (dpsi/deta) evaluated at given points
	#         Arrays are of shape (num_nodes, num_points)
	def evaluate_dpsi(self, local_coordinates):
		num_points = local_coordinates.shape[0]
		num_nodes = (self.p + 1)**2
		dpsi0 = np.zeros((num_nodes, num_points), dtype = np.float32) # dxi
		dpsi1 = np.zeros((num_nodes, num_points), dtype = np.float32) # deta
		
		if (self.p == 1):
			psi_1d_xi   = self.__psi_p1(local_coordinates[:,0])
			psi_1d_eta  = self.__psi_p1(local_coordinates[:,1])
			dpsi_1d_xi  = self.__dpsi_p1(local_coordinates[:,0])
			dpsi_1d_eta = self.__dpsi_p1(local_coordinates[:,1])
		elif (self.p == 2):
			psi_1d_xi   = self.__psi_p2(local_coordinates[:,0])
			psi_1d_eta  = self.__psi_p2(local_coordinates[:,1])
			dpsi_1d_xi  = self.__dpsi_p2(local_coordinates[:,0])
			dpsi_1d_eta = self.__dpsi_p2(local_coordinates[:,1])
		else:
			error_msg = "ERROR: element not available for p = " + str(self.p)
			sys.exit(error_msg)
		# if	
		
		# Calculate 2D shape function derivatives as products of 
		# 1D shape functions (and derivatives)
		# TODO: vectorize (outer product and flatten)
		for j in range(num_points):
			i = 0;
			for i_eta in range(self.p + 1):
				for i_xi in range(self.p + 1):
					dpsi0[i,j] = psi_1d_eta[i_eta,j]*dpsi_1d_xi[i_xi,j]
					dpsi1[i,j] = dpsi_1d_eta[i_eta,j]*psi_1d_xi[i_xi,j]
					i += 1
				# for i_xi
			# for i_eta
		# for j
		
		return dpsi0, dpsi1

	# def evaluate_dpsi
	
	# Evaluate 1D p=1 shape functions at points xi
	# Length of array xi is num_points 
	# Shape of returned array is (2, num_points)
	def __psi_p1(self, xi):
		num_points = len(xi)
		psi = np.zeros((2, num_points), dtype = np.float32)
		psi[0, :] = 0.5*(1 - xi)
		psi[1, :] = 0.5*(1 + xi)
		return psi
	# def __psi_p1
	
	# Evaluate 1D p=1 shape function derivatives at points xi
	# Length of array xi is num_points 
	# Shape of returned array is (2, num_points)
	def __dpsi_p1(self, xi):
		num_points = len(xi)
		psi = np.zeros((2, num_points), dtype = np.float32)
		psi[0, :] = -0.5
		psi[1, :] = 0.5
		return psi
	# def __dpsi_p1
	
	# Evaluate 1D p=2 shape functions at points xi
	# Length of array xi is num_points 
	# Shape of returned array is (3, num_points)
	def __psi_p2(self, xi):
		num_points = len(xi)
		psi = np.zeros((3, num_points), dtype = np.float32)
		psi[0, :] = 0.5*xi*(xi - 1)
		psi[1, :] = (1 - xi)*(1 + xi)
		psi[2, :] = 0.5*xi*(xi + 1)
		return psi
	# def __psi_p2
	
	# Evaluate 1D p=2 shape function derivatives at points xi
	# Length of array xi is num_points 
	# Shape of returned array is (3, num_points)
	def __dpsi_p2(self, xi):
		num_points = len(xi)
		psi = np.zeros((3, num_points), dtype = np.float32)
		psi[0, :] = xi - 0.5
		psi[1, :] = - 2.0*xi
		psi[2, :] = xi + 0.5
		return psi
	# def __dpsi_p2

# class Element	
