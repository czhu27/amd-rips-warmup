import numpy as np
import math
import os

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from element import Element2d
from quadrature import Gauss2d

# ------------------------------------------------------------------------------
# class Field (contains variables for given spatial discretization)
# ------------------------------------------------------------------------------
class Field:
	# Constructor
	def __init__(self, params, mesh):
		self.ps = params["ps"]
		self.figs_dir = params["data_dir"] + "/figs"
		self.traces_dir = params["data_dir"] + "/traces"
		os.makedirs(self.figs_dir)
		os.makedirs(self.traces_dir)

		self.num_nodes_per_element = (self.ps + 1)*(self.ps + 1)
		self.mesh = mesh
		if "dt" in params:
			self.dt = params["dt"]
		else:	# sometimes missing when params are specified for tests
			self.dt = 0.0	
		self.__define_global_coordinates()	# of computational nodes
		self.__determine_edge_indices() # for flux computation

		# Handle sources
		if "src_loc" in params:
			self.src_loc = np.array(params["src_loc"])
			self.src_f = params["src_f"]
			self.src_a = params["src_a"]
		else:	# no source!
			self.src_loc = np.array([[0, 0]])
			self.src_f = 1.0
			self.src_a = 0.0
		self.setup_sources()

		# Handle receivers
		if "rcv_loc" in params:
			self.has_receivers = True
			self.rcv_loc = np.array(params["rcv_loc"])
			num_steps = math.ceil(params["tf"]/params["dt"])
			self.setup_receivers(num_steps)	
		else:
			self.has_receivers = False
		

	# def __init__
	
	def __define_global_coordinates(self):
		ne = self.mesh.ne
		nn = self.num_nodes_per_element
		self.global_node_coordinates = np.zeros((ne, 2*nn), dtype = np.float32)

		# New up element p to get local coordinates (num_coordinates x 2)
		element = Element2d(self.ps)
		local_coordinates = element.coordinates

		# New up element p = 1 to evaluate shape function at local coordinates
		# psi is of shape (4 x num_coordinates) -- 4 shape functions for 
		# bilinear element
		element = Element2d(1)
		psi = element.evaluate_psi(local_coordinates)

		# Calculate global coordinates (for each element)
		for e in range(ne):
			# Combine x coordinates of element vertices in single array
			# x occupies the 0th, 2th, 4th, and 6th position
			x = self.mesh.element_vertex_coordinates[e][0::2] 
			y = self.mesh.element_vertex_coordinates[e][1::2] 

			for p in range(nn):
				self.global_node_coordinates[e, 2*p + 0] = np.dot(x, psi[:, p])	
				self.global_node_coordinates[e, 2*p + 1] = np.dot(y, psi[:, p])	
			# for p
		# for e
	# def __define_global_coordinates

	def __determine_edge_indices(self):
		num_elements = self.mesh.ne
		nex = self.mesh.nx
		ney = self.mesh.ny
		num_edges = 4
		num_nodes_per_edge = self.ps + 1
		self.edge_node_ids_in = np.zeros((num_elements, 
										  num_edges, 
										  num_nodes_per_edge), dtype=int)
		self.edge_node_ids_ex = np.zeros((num_elements, 
										  num_edges, 
										  num_nodes_per_edge), dtype=int)
		
		# External field to be multiplied by one by default, expect when
		# edge is on boundary.
		self.edge_pex_factor = np.ones((num_elements,
									    num_edges,
									    num_nodes_per_edge), dtype=int)								  
		self.edge_uex_factor = np.ones((num_elements,
									    num_edges,
									    num_nodes_per_edge), dtype=int)								  
		self.edge_vex_factor = np.ones((num_elements,
									    num_edges,
									    num_nodes_per_edge), dtype=int)								  
		for e in range(num_elements):
			neighbor_ids = self.mesh.element_neighbor_ids[e]

			# South (ym)
			for i in range(num_nodes_per_edge):
				# IN
				node_id = i
				self.edge_node_ids_in[e, 0, i] = node_id
				# EX
				node_id = (num_nodes_per_edge - 1)*num_nodes_per_edge + i 
				self.edge_node_ids_ex[e, 0, i] = node_id
			# for i	
			
			if (e == neighbor_ids[0]):	# South edge is boundary
				self.edge_node_ids_ex[e, 0, :] = self.edge_node_ids_in[e, 0, :]
				self.edge_pex_factor[e, 0, :] = 1.0
				self.edge_uex_factor[e, 0, :] = -1.0
				self.edge_vex_factor[e, 0, :] = -1.0

			# East (xp)
			for i in range(num_nodes_per_edge):
				# IN
				node_id = i*num_nodes_per_edge + num_nodes_per_edge - 1
				self.edge_node_ids_in[e, 1, i] = node_id
				#EX
				node_id = i*num_nodes_per_edge 
				self.edge_node_ids_ex[e, 1, i] = node_id
			# for i	
			
			if (e == neighbor_ids[1]):
				self.edge_node_ids_ex[e, 1, :] = self.edge_node_ids_in[e, 1, :]
				self.edge_pex_factor[e, 1, :] = 1.0
				self.edge_uex_factor[e, 1, :] = -1.0
				self.edge_vex_factor[e, 1, :] = -1.0

			# North (yp)
			for i in range(num_nodes_per_edge):
				# IN
				node_id = (num_nodes_per_edge - 1)*num_nodes_per_edge + i 
				self.edge_node_ids_in[e, 2, i] = node_id
				#EX
				node_id = i
				self.edge_node_ids_ex[e, 2, i] = node_id
			# for i	
			
			if (e == neighbor_ids[2]):
				self.edge_node_ids_ex[e, 2, :] = self.edge_node_ids_in[e, 2, :]
				self.edge_pex_factor[e, 2, :] = 1.0
				self.edge_uex_factor[e, 2, :] = -1.0
				self.edge_vex_factor[e, 2, :] = -1.0

			# West (xm)
			for i in range(num_nodes_per_edge):
				# IN
				node_id = i*num_nodes_per_edge 
				self.edge_node_ids_in[e, 3, i] = node_id
				# EX
				node_id = i*num_nodes_per_edge + num_nodes_per_edge - 1
				self.edge_node_ids_ex[e, 3, i] = node_id
			# for i	
			
			if (e == neighbor_ids[3]):
				self.edge_node_ids_ex[e, 3, :] = self.edge_node_ids_in[e, 3, :]
				self.edge_pex_factor[e, 3, :] = 1.0
				self.edge_uex_factor[e, 3, :] = -1.0
				self.edge_vex_factor[e, 3, :] = -1.0
		# for e

	# def __dtermine_edge_indices

	def create_fields_acoustic(self):
		ne = self.mesh.ne
		nn = self.num_nodes_per_element
		self.val = {}
		self.val["p"] = np.zeros((ne, nn), dtype = np.float32)
		self.val["u"] = np.zeros((ne, nn), dtype = np.float32)
		self.val["v"] = np.zeros((ne, nn), dtype = np.float32)

		self.rhs = {}
		self.rhs["p"] = np.zeros((ne, nn), dtype = np.float32)
		self.rhs["u"] = np.zeros((ne, nn), dtype = np.float32)
		self.rhs["v"] = np.zeros((ne, nn), dtype = np.float32)

		print("Acoustic fields (p,u,v) have been created.")
	# def __create_fields
	
	def __expression_cos(self, x, y, t):
		n_x = 2
		n_y = 2
		n_t = np.sqrt(n_x*n_x + n_y*n_y)
		U0 = - n_x/n_t
		V0 = - n_y/n_t
		p =    np.cos(n_x*np.pi*x)*np.cos(n_y*np.pi*y)*np.sin(n_t*np.pi*t)
		u = U0*np.sin(n_x*np.pi*x)*np.cos(n_y*np.pi*y)*np.cos(n_t*np.pi*t)
		v = V0*np.cos(n_x*np.pi*x)*np.sin(n_y*np.pi*y)*np.cos(n_t*np.pi*t)
		return p, u, v
	# def __expression_cos

	def __expression_gaussian(self, x, y, t):
		#r2 = (x - 0.5)**2 + (y - 0.5)**2
		r2 = (x - 0.5)**2 + (y - 0.5)**2
		p = np.exp(-30.0*r2)
		u = np.zeros(x.shape)
		v = np.zeros(x.shape)
		return p, u, v

	# def __expression_gaussian

	# Following is temporary until more generic approach is implemented
	def initialize(self, initializer):

		if (initializer == 'cos'):
			fn = self.__expression_cos
		elif (initializer == 'gaussian'):
			fn = self.__expression_gaussian
		elif (initializer == 'zero'):
			return	# don't do anything -- all fields are zero
		else:
			# keep everything at zero
			print('   Unknown initializer function.  ' + 
				  'Setting all fields to zero.')
			return

		nn = self.num_nodes_per_element
		for n in range(nn):
			x = self.global_node_coordinates[:, 2*n + 0]
			y = self.global_node_coordinates[:, 2*n + 1]
			p, u, v = fn(x, y, 0.0)
			self.val["p"][:, n] = p
			self.val["u"][:, n] = u
			self.val["v"][:, n] = v
		# for n
		print("Fields (p,u,v) have been initialized.")
	# def initialize_p

	# Following is temporary until more generic approach is implemented
	def display_p(self, index):
		nx = self.mesh.nx # number of elements along x axis
		ny = self.mesh.ny # number of elements along y axis
		npts = self.ps + 1 # number of nodes on one side of element
		num_nodes_x = nx*npts # total number of nodes along x axis
		num_nodes_y = ny*npts # total number of nodes along y axis
		
		# Define grid for displaying field
		xg = np.zeros((num_nodes_x, num_nodes_y), dtype = np.float32)
		yg = np.zeros((num_nodes_x, num_nodes_y), dtype = np.float32)
		pg = np.zeros((num_nodes_x, num_nodes_y), dtype = np.float32)
		for j in range(ny):
			for jj in range(npts):
				idx_j = j*npts + jj
				for i in range(nx):
					for ii in range(npts):
						idx_i = i*npts + ii
						e = j*nx + i
						node_index = jj*npts + ii
						x = self.global_node_coordinates[e, 2*node_index + 0]
						y = self.global_node_coordinates[e, 2*node_index + 1]
						p = self.val["p"][e, node_index]
						xg[idx_i, idx_j] = x;
						yg[idx_i, idx_j] = y;
						pg[idx_i, idx_j] = p;
					# for ii
				# for i
			# for jj
		# for j	
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(xg, yg, pg, cmap=cm.coolwarm)
		ax.set_title("Simulation (t = {:.2F})".format(self.dt*index))
		#plt.show()
		plt.tight_layout(pad=0.75)
		#str_name = "wave/forward/figs/fig_sim_{:03d}.png".format(index)
		str_name = self.figs_dir + "/fig_sim_{:03d}.png".format(index)
		plt.savefig(str_name)
		plt.close()

	# def display_p

	def display_diagnostics(self):
		min_max = np.zeros((len(self.val) ,2))
		idx = 0
		for f in self.val:
			max_val = np.max(self.val[f])
			min_val = np.min(self.val[f])
			min_max[idx, 0] = min_val
			min_max[idx, 1] = max_val
			print(f + " in [" + "{:.15E}".format(min_val) + ", " 
					+ "{:.15E}".format(max_val) + "]")
			idx += 1		
		return min_max	
	# def display_diagnostics

	def compute_errors(self, params):
		# High-order quadrature rule for error computation
		gauss = Gauss2d(params["int_diag"])
		tf = params["tf"]
		initializer = params["initializer"]
		
		if (initializer == 'cos'):
			fn = self.__expression_cos
		else:
			# keep everything at zero
			return

		# Retrieve fields at nodal locations
		f_p = self.val["p"]
		f_u = self.val["u"]
		f_v = self.val["v"]
		
		# Interpolative element
		element = Element2d(self.ps)
		psi = element.evaluate_psi(gauss.coordinates)

		# Compute field values at quadrature points
		ph = np.dot(f_p, psi)
		uh = np.dot(f_u, psi)
		vh = np.dot(f_v, psi)
		
		self.mesh.compute_isomorphisms(gauss.coordinates)
		iso = self.mesh.isomorphisms
		global_coordinates = self.mesh.global_coordinates
		pe = np.zeros((self.mesh.ne, gauss.num_points))
		ue = np.zeros((self.mesh.ne, gauss.num_points))
		ve = np.zeros((self.mesh.ne, gauss.num_points))
	
		for n in range(gauss.num_points):
			x = global_coordinates[:, 2*n + 0]
			y = global_coordinates[:, 2*n + 1]
			pe[:, n], ue[:, n], ve[:, n] = fn(x, y, tf)
		# for n
		
		diff_p = (ph - pe)**2
		diff_u = (uh - ue)**2
		diff_v = (vh - ve)**2
		err_p = np.sqrt(np.sum(np.dot(diff_p*iso[:,:,0], gauss.weights)))
		err_u = np.sqrt(np.sum(np.dot(diff_u*iso[:,:,0], gauss.weights)))
		err_v = np.sqrt(np.sum(np.dot(diff_v*iso[:,:,0], gauss.weights)))
		print("Error on p: " + "{:.6E}".format(err_p))
		print("Error on u: " + "{:.6E}".format(err_u))
		print("Error on v: " + "{:.6E}".format(err_v))
	# def compute_errors

	# Set up source term (find elements and compute shape functions)
	def setup_sources(self):
		self.src_eid, local_coordinates = self.mesh.find_elements(self.src_loc)
		element = Element2d(self.ps) 
		self.src_psi = element.evaluate_psi(local_coordinates)

	# Setup receivers (find elements and compute shape functions)	
	def setup_receivers(self, num_steps):
		if self.has_receivers: 
			self.rcv_eid, local_coordinates = \
									self.mesh.find_elements(self.rcv_loc)
			element = Element2d(self.ps)
			self.rcv_psi = element.evaluate_psi(local_coordinates)
			
			# Allocate space to record receiver traces
			num_rcvs = self.rcv_loc.shape[0]
			# One trace per receiver + one trace for the timeline
			self.traces = np.zeros((num_rcvs + 1, num_steps))
	
	# Save all traces to disk
	def save_traces(self):
		if self.has_receivers:
			#fname = "wave/forward/traces/traces.npz"
			fname = self.traces_dir + "/traces.npz"
			f = open(fname, "w")
			np.savez(fname, traces = self.traces)
			f.close()

# class Field
