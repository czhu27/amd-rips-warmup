import numpy as np
import math
from element import Element2d

# Element geometry and numbering
#
#  v2        e2        v3
#    *----------------*
#    |6      7       8|
#    |                |
#    |                |
# e3 |3      4       5| e1
#    |                |
#    |                |
#    |0      1       2|
#    *----------------*
#  v0        e0        v1
# 
# . Numbering for neighbors follows the same convention as for edges
# . Numbers inside the element show the numbering for computational nodes
#   (here for p = 2).
# . Vertex numbering follows the convention used for node numbering of p = 1 
#   element

# ------------------------------------------------------------------------------
# class Mesh
# ------------------------------------------------------------------------------
class Mesh:
	# Constructor
	def __init__(self, params):
		self.x0 = float(params["x0"])
		self.y0 = float(params["y0"])
		self.x1 = float(params["x1"])
		self.y1 = float(params["y1"])
		self.lx = self.x1 - self.x0
		self.ly = self.y1 - self.y0
		self.nx = int(params["nx"])
		self.ny = int(params["ny"])
		self.ps = int(params["ps"])
		self.pm = int(params["pm"])
		self.__build()	# build mesh (create vertices, etc.)
	# def __init__	

	# Build mesh
	def __build(self):
		# Calculate/store number of elements
		self.hx = float(self.lx/self.nx)
		self.hy = float(self.ly/self.ny)
		self.ne = self.nx*self.ny

		# Define element geometries
		# For given element, store vertex coordinates as follows:
		# [x0 y0 x1 y1 x2 y2 x3 y3]
		self.element_vertex_coordinates = np.zeros((self.ne, 8), 
												   dtype=np.float32)
		e = 0
		for j in range(self.ny):
			for i in range(self.nx):
				x0 = i*self.hx
				y0 = j*self.hy
				self.element_vertex_coordinates[e][0] = x0
				self.element_vertex_coordinates[e][1] = y0
				self.element_vertex_coordinates[e][2] = x0 + self.hx
				self.element_vertex_coordinates[e][3] = y0
				self.element_vertex_coordinates[e][4] = x0
				self.element_vertex_coordinates[e][5] = y0 + self.hy
				self.element_vertex_coordinates[e][6] = x0 + self.hx 
				self.element_vertex_coordinates[e][7] = y0 + self.hy
				e += 1
			# for i
		# for j

		# Determine neighbors (if no neighbor, neighbor Id = own Id)
		self.element_neighbor_ids = np.zeros((self.ne, 4), 
											 dtype=int)
		e = 0
		for j in range(self.ny):
			for i in range(self.nx):
				n0 = (e if (j == 0) else e - self.nx)
				self.element_neighbor_ids[e][0] = n0 
				
				n1 = (e if (i == self.nx - 1) else e + 1)
				self.element_neighbor_ids[e][1] = n1
				
				n2 = (e if (j == self.ny - 1) else e + self.nx)
				self.element_neighbor_ids[e][2] = n2
				
				n3 = (e if (i == 0) else e - 1)
				self.element_neighbor_ids[e][3] = n3
				e += 1
			# for i
		# for j

		# Compute edge jacobians and normals (constant per edge).  Edge normals
		# are stored as follows: n0_x, n0_y, n1_x, n1_y, ...
		self.edge_jacobians = np.zeros((self.ne, 4), dtype=np.float32)
		self.element_normals = np.zeros((self.ne, 8), dtype=np.float32)
		for e in range(self.ne):
			# South
			x0 = self.element_vertex_coordinates[e][0]
			y0 = self.element_vertex_coordinates[e][1]
			x1 = self.element_vertex_coordinates[e][2]
			y1 = self.element_vertex_coordinates[e][3]
			l = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
			self.edge_jacobians[e][0] = 0.5*l
			self.element_normals[e][0] = (y1 - y0)/l
			self.element_normals[e][1] = (x0 - x1)/l
			
			# East
			x0 = self.element_vertex_coordinates[e][2]
			y0 = self.element_vertex_coordinates[e][3]
			x1 = self.element_vertex_coordinates[e][6]
			y1 = self.element_vertex_coordinates[e][7]
			l = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
			self.edge_jacobians[e][1] = 0.5*l
			self.element_normals[e][2] = (y1 - y0)/l
			self.element_normals[e][3] = (x0 - x1)/l

			# North
			x0 = self.element_vertex_coordinates[e][6]
			y0 = self.element_vertex_coordinates[e][7]
			x1 = self.element_vertex_coordinates[e][4]
			y1 = self.element_vertex_coordinates[e][5]
			l = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
			self.edge_jacobians[e][2] = 0.5*l
			self.element_normals[e][4] = (y1 - y0)/l
			self.element_normals[e][5] = (x0 - x1)/l

			# West
			x0 = self.element_vertex_coordinates[e][4]
			y0 = self.element_vertex_coordinates[e][5]
			x1 = self.element_vertex_coordinates[e][0]
			y1 = self.element_vertex_coordinates[e][1]
			l = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
			self.edge_jacobians[e][3] = 0.5*l
			self.element_normals[e][6] = (y1 - y0)/l
			self.element_normals[e][7] = (x0 - x1)/l
		
	# def build

	# Compute all element Jacobians and inverse maps at given local points
	# local_coordinates: array is of shape (num_points, 2)
	# Store in 3D array isomorphisms of shape num_elements x num_points x 5,
	# where the '5' comes from [jacobian dxi/dx dxi/dy deta/dx deta/dy]
	# Also store global coordinates corresponding to local points in each 
	# element.
	def compute_isomorphisms(self, local_coordinates):
		num_points = local_coordinates.shape[0]

		# Create P1 element (consider bilinear isomorphism)
		element = Element2d(1)

		# Compute shape function values at local coordinates
		# psi is of shape (4, num_points)
		psi = element.evaluate_psi(local_coordinates)

		# Compute shape function derivatives w.r.t. local coordinates
		# dpsi0 and dpsi1 are both of shape (4, num_points)
		dpsi0, dpsi1 = element.evaluate_dpsi(local_coordinates)

		# Isomorphisms
		self.isomorphisms = np.zeros((self.ne, num_points, 5), 
									 dtype = np.float32)

		# Global coordinates (per element: [x0 y0 x1 y1 ...])
		self.global_coordinates = np.zeros((self.ne, 2*num_points),
										   dtype = np.float32)
		
		# For each element, compute all required isomorphisms
		for e in range(self.ne):
			# Combine x coordinates of element vertices in single array
			# x occupies the 0th, 2th, 4th, and 6th position
			x = self.element_vertex_coordinates[e][0::2] 
			y = self.element_vertex_coordinates[e][1::2] 

			# Element vertices are defined in a clockwise fashion,
			# starting with the bottom left corner.  However, the shape
			# functions are defined line by line.  Therefore, the x
			# and y coordinates have to be reordered to be aligned
			# with the shape functions
			#x = np.array([xv[0], xv[1], xv[3], xv[2]])
			#y = np.array([yv[0], yv[1], yv[3], yv[2]])

			# Go over all points where shape derivatives have been computed
			for p in range(num_points):
				# Global coordinates
				self.global_coordinates[e, 2*p + 0] = np.dot(x, psi[:, p])
				self.global_coordinates[e, 2*p + 1] = np.dot(y, psi[:, p])
				
				# Isomorphism
				dxdxi  = np.dot(x, dpsi0[:, p])
				dxdeta = np.dot(x, dpsi1[:, p])
				dydxi  = np.dot(y, dpsi0[:, p])
				dydeta = np.dot(y, dpsi1[:, p])
				jacobian = dxdxi*dydeta - dxdeta*dydxi
				dxidx =    dydeta/jacobian
				dxidy =  - dxdeta/jacobian
				detadx = - dydxi/jacobian
				detady =   dxdxi/jacobian
				self.isomorphisms[e][p][0] = jacobian
				self.isomorphisms[e][p][1] = dxidx
				self.isomorphisms[e][p][2] = dxidy
				self.isomorphisms[e][p][3] = detadx
				self.isomorphisms[e][p][4] = detady
			# for p
		# for e
	# def compute_isomorphisms

	# Helper function.  Find element that owns given global coordinates.
	# Return element ids and associated local coordinates
	# global_coordinates is of shape (num_points, 2)
	def find_elements(self, global_coordinates):
		num_points = global_coordinates.shape[0]

		element_ids = np.zeros(num_points, dtype=int)
		local_coordinates = np.zeros((num_points, 2), dtype=np.float32)
		
		for i in range(num_points):
			x = global_coordinates[i, 0]
			y = global_coordinates[i, 1]
			
			# Find element that owns the source
			src_i = math.floor((x - self.x0)/self.hx)
			if src_i > self.nx - 1:
				src_i -= 1
			
			src_j = math.floor((y - self.y0)/self.hy)
			if src_j > self.ny - 1:
				src_j -= 1
			
			element_ids[i] = src_j*self.nx + src_i
		
			# Compute local coordinates of source
			x0 = self.element_vertex_coordinates[element_ids[i]][0]
			xi0 = 2*(x - x0)/self.hx - 1.0  
			y0 = self.element_vertex_coordinates[element_ids[i]][1]
			xi1 = 2*(y - y0)/self.hy - 1.0  
			local_coordinates[i, 0] = xi0
			local_coordinates[i, 1] = xi1

		# def fine_elements	
		return element_ids, local_coordinates

# class Mesh
