import numpy as np
import os
import math

from element import Element2d
from quadrature import GLL2d
from quadrature import Gauss2d

# When flag is True, field values are dumped to files every time step
__ML_DUMP__ = True

# ------------------------------------------------------------------------------
# class KernelAcoustic
# ------------------------------------------------------------------------------
class KernelAcoustic:
	def __init__(self, mesh, field, params):
		self.mesh = mesh
		self.field = field
		self.dumps_dir = params["data_dir"] + "/dumps"
		os.makedirs(self.dumps_dir)
		self.data_percents = params["data_percents"]
		self.num_steps = math.ceil(1/params["dt"])	
		self.dt = params["dt"]


		# Create element and quadrature (note that the quadrature points are
		# colocated with the element coordinates)
		self.__element = Element2d(self.field.ps) 
		self.__quadrature = GLL2d(self.field.ps + 1)

		# Precompute all element isomorphisms (jacobians and maps)
		self.mesh.compute_isomorphisms(self.__quadrature.coordinates)

		# Precompute shape function values and derivatives at quadrature points
		# psi, dpsi0 (dpsi/dxi), and dpsi1 (dpsi/deta) are of shape
		# (num_shape_functions, num_integration_points)
		coords = self.__quadrature.coordinates
		self.__psi = self.__element.evaluate_psi(coords)
		self.__dpsi0, self.__dpsi1 = self.__element.evaluate_dpsi(coords)
	
		# Used for tracking snapshots dumped for ML training
		self.__ml_dump_index = 0
		#NEED TO ASSERT THIS WORKS SOMEWHERE
		if "sample_step" in params:
			step_size = int(params["sample_step"]/params["dt"])
			assert params["sample_step"]/params["dt"] - step_size < 0.001, "Sample step not a multiple of dt"
			self.ml_dump_step = step_size
		else:
			self.ml_dump_step = 1		

		# Used to keep track of time
		self.t = 0.0;

	# Compute mass matrices
	def compute_mass(self):
		iso = self.mesh.isomorphisms
		
		# Mass matrix is diagonal!
		self.__mass = np.zeros((self.mesh.ne, self.__element.num_nodes))

		# Evaluate psi_i*psi_j at all integration points and integrate the 
		# product in each element
		for i in range(self.__psi.shape[0]):
			for j in range(self.__psi.shape[0]):
				# Compute product of shape functions at all integration points
				psi_psi = self.__psi[i,:]*self.__psi[j,:]
				# Use broadcasting to multiply product of shape functions by 
				# jacobian.  Dot product with integration weights gives integral
				# of psi_i * psi_j on all elements at once
				int_psi_psi = np.dot(psi_psi*iso[:,:,0], 
									 self.__quadrature.weights)
				self.__mass[:,i] += int_psi_psi
		# ! Note that self.__mass[e,i] is the integral of psi_i * psi_i 
		# on element e.  Once the rhs associated with shape function psi_i
		# is computed on element e, it has to be divided by self.__mass[:,i]

	# Set all right-hand sides to 0
	def reset(self):
		self.field.rhs["p"].fill(0.0)
		self.field.rhs["u"].fill(0.0)
		self.field.rhs["v"].fill(0.0)
	# def reset

	# Compute volume term - add to right-hand side
	def compute_volume(self):
		# Retrieve all element isomorphisms
		# iso shape is (num_elements, num_integration_points, 5), where 5
		# is for [jacobian, dxidx, dxidy, detadx, detady]
		iso = self.mesh.isomorphisms

		# Retrieve field nodal values (num_elements, num_nodes_per_element)
		f_p = self.field.val["p"]
		f_u = self.field.val["u"]
		f_v = self.field.val["v"]
		rhs_p = self.field.rhs["p"]
		rhs_u = self.field.rhs["u"]
		rhs_v = self.field.rhs["v"]
		
		# ===== Pressure (p) equation =====
		# For each element, vol_i = - < \kappa \psi_i \div u >
		kappa = 1.0	# Assume kappa is one for now
		
		# Compute velocity divergence at all integration points, in each element
		# f_u is (num_elements, num_nodes_per_element) and 
		# dpsi0 is (num_nodes_per_element, num_integration_points) 
		# dudx is (num_elements, num_integration_points)
		dudx = np.dot(f_u, self.__dpsi0)*iso[:,:,1] + \
			   np.dot(f_u, self.__dpsi1)*iso[:,:,3]
		dvdy = np.dot(f_v, self.__dpsi0)*iso[:,:,2] + \
			   np.dot(f_v, self.__dpsi1)*iso[:,:,4]
		# FLOPS
		# Ne = # number of elements
		# NNe = # nodes per element
		# Nk = # integration points
		# for dudx : 2*(Ne*Nk*(2NNe - 1) + Ne*Nk) = 4*Ne*Nk*NNe
		# Same for dvdy
		
		# For each shape function (evaluated at all integration points),
		# compute the product psi_i * div u, and integrate over each element
		# product_i is of shape (num_elements, num_integration_points)
		for i in range(self.__psi.shape[0]):
			product = - kappa*(dudx + dvdy)*self.__psi[i,:]
			# FLOPS for product
			# 3*Ne*Nk
			integral = np.dot(product*iso[:,:,0], self.__quadrature.weights)
			# FLOPS for integral
			# Ne*Nk + Ne*(2*Nk - 1) = 2*Ne*Nk	
			rhs_p[:, i] += integral			 
			# FLOPS for rhs_p
			# Ne
		# FLOPS - Total for loop
		# NNe*(3*Ne*Nk + 2*Ne*Nk + Ne) = NNe*(5*Ne*Nk + Ne)
		
		# ===== Velocity (u,v) equations =====
		# For each element, compute - <(1/\rho) \psi_i dpdx>
		inv_rho = 1.0	# assume rho = 1 for now

		# Compute dpdx at all integration points, in each element
		dpdx = np.dot(f_p, self.__dpsi0)*iso[:,:,1] + \
			   np.dot(f_p, self.__dpsi1)*iso[:,:,3]
		dpdy = np.dot(f_p, self.__dpsi0)*iso[:,:,2] + \
			   np.dot(f_p, self.__dpsi1)*iso[:,:,4]

		# For each shape function (evaluated at all integration points),
		# compute the products \psi_i dpdx and \psi_i dpdy
		for i in range(self.__psi.shape[0]):
			product_u = - inv_rho*dpdx*self.__psi[i,:]
			product_v = - inv_rho*dpdy*self.__psi[i,:]
			integral_u = np.dot(product_u*iso[:,:,0], self.__quadrature.weights)
			integral_v = np.dot(product_v*iso[:,:,0], self.__quadrature.weights)
			rhs_u[:,i] += integral_u
			rhs_v[:,i] += integral_v
		
	# Compute fluxes
	def compute_fluxes(self):
		# Retrieve field nodal values (num_elements, num_nodes_per_element)
		f_p = self.field.val["p"]
		f_u = self.field.val["u"]
		f_v = self.field.val["v"]
		rhs_p = self.field.rhs["p"]
		rhs_u = self.field.rhs["u"]
		rhs_v = self.field.rhs["v"]
			
		#for e in range(self.mesh.ne):
		kappa_in = 1.0
		inv_rho_in = 1.0
		z_in = 1.0
		z_ex = 1.0
		sum_z = z_in + z_ex
		inv_sum_z = 1.0/sum_z
		
		# Retrieve normals.  Order is n0_x, n0_y, n1_x, n1_y, ...
		nx = self.mesh.element_normals[:, [0, 2, 4, 6]]
		ny = self.mesh.element_normals[:, [1, 3, 5, 7]]

		# Loop over edges (all elements for given edge!)
		for k in range(4):
			
			# Get in and ex node indices for current edge (all elements!)
			idx_in = self.field.edge_node_ids_in[:, k]
			idx_ex = self.field.edge_node_ids_ex[:, k]
			p_ex_factor = self.field.edge_pex_factor[:, k]
			u_ex_factor = self.field.edge_uex_factor[:, k]
			v_ex_factor = self.field.edge_vex_factor[:, k]
			
			p_in = np.take_along_axis(f_p, idx_in, axis = 1)
			u_in = np.take_along_axis(f_u, idx_in, axis = 1)
			v_in = np.take_along_axis(f_v, idx_in, axis = 1)

			# Vector of length 'num_element' that contains id of
			# neighbor across current edge
			neighbor_id = self.mesh.element_neighbor_ids[:, k]
			
			# Precompute factor for each node (BCs).  If neighbor
			# does not exist, neighbor id is self and idx_ex is equal to
			# idx_in.  Multipliplying factor must then be introduced.
			p_ex = np.take_along_axis(f_p[neighbor_id, :], idx_ex, axis = 1)
			u_ex = np.take_along_axis(f_u[neighbor_id, :], idx_ex, axis = 1)
			v_ex = np.take_along_axis(f_v[neighbor_id, :], idx_ex, axis = 1)
			
			p_ex *= p_ex_factor;
			u_ex *= u_ex_factor;
			v_ex *= v_ex_factor;

			nx_k = (nx[:, k])[:, np.newaxis]
			ny_k = (ny[:, k])[:, np.newaxis]
			vn_jump = ((u_in - u_ex)*nx_k + (v_in - v_ex)*ny_k)
			p_jump = (p_in - p_ex)
			'''
			if (k == 0):
				fname = 'jump_time_series/jump_t.txt'
				f = open(fname, "a")
				out = np.array([self.t])
				np.savetxt(f, out, fmt='%.8e')
				f.close()
				fname = 'jump_time_series/jump_p.txt'
				f = open(fname, "a")
				out = np.array([p_jump[511,1]])
				np.savetxt(f, out, fmt='%.8e')
				f.close()
			'''
			# Compute flux function value at edge integration points
			flux_p = inv_sum_z*kappa_in*(z_ex*vn_jump - p_jump)
			flux_u = inv_sum_z*z_in*nx_k*inv_rho_in*(p_jump - z_ex*vn_jump)
			flux_v = inv_sum_z*z_in*ny_k*inv_rho_in*(p_jump - z_ex*vn_jump)
			
			# Multiply by quadrature weights and edge jacobian
			jac_k = (self.mesh.edge_jacobians[:, k])[:, np.newaxis]

			flux_p *= self.__quadrature.weights_1d*jac_k
			flux_u *= self.__quadrature.weights_1d*jac_k
			flux_v *= self.__quadrature.weights_1d*jac_k
		
			# Update right-hand sides
			rhs_p_new = np.take_along_axis(rhs_p, idx_in, axis = 1) + flux_p
			rhs_u_new = np.take_along_axis(rhs_u, idx_in, axis = 1) + flux_u
			rhs_v_new = np.take_along_axis(rhs_v, idx_in, axis = 1) + flux_v
			np.put_along_axis(rhs_p, idx_in, rhs_p_new, axis = 1)	
			np.put_along_axis(rhs_u, idx_in, rhs_u_new, axis = 1)	
			np.put_along_axis(rhs_v, idx_in, rhs_v_new, axis = 1)	
			
		# for k
	# def compute_fluxes
	
	# Scale by mass matrix
	def wrapup(self):
		for f in self.field.rhs:
			self.field.rhs[f] = np.copy(np.divide(self.field.rhs[f], 
										self.__mass))	

	def compute_sources(self, t):
		# Compute time signature
		freq = self.field.src_f
		t0 = 1.0/freq
		f2 = np.square(2*np.pi*freq)
		signature = (1 - f2*np.square(t - t0))*np.exp(-0.5*f2*np.square(t - t0))

		# Apply sources in elements that own it
		rhs_p = self.field.rhs["p"]
		for i in range(self.field.src_eid.shape[0]):

			eid = self.field.src_eid[i]
			psi = self.field.src_psi[:, i] # shape is (num_shape_fns, num_pts)
			rhs_p[eid, :] += self.field.src_a*signature*psi

	# Record pressure value at receiver location.  Save in trace.
	def record_receivers(self, step_number, t):
		if self.field.has_receivers:
			rhs_p = self.field.val["p"]
			num_rcvs = self.field.rcv_eid.shape[0]
			for i in range(self.field.rcv_eid.shape[0]):
				eid = self.field.rcv_eid[i]
				psi = self.field.rcv_psi[:, i]
				p = np.dot(psi, rhs_p[eid, :])
				self.field.traces[i, step_number] = p
			self.field.traces[num_rcvs, step_number] = t

	def clean_data(self):
		#Flattens all arrays to make indexing easier
		p = np.ndarray.flatten(self.field.val["p"])
		u = np.ndarray.flatten(self.field.val['u'])
		v = np.ndarray.flatten(self.field.val['v'])
		xy = np.ndarray.flatten(self.field.global_node_coordinates)
		x = xy[0::2]
		y = xy[1::2]

		#Separates boundary points for all variables
		indices = np.argwhere((x == 0) | (x == 1) | (y == 0) | (y == 1))
		boundaries = np.zeros((len(indices), 6))
		boundaries[:,0] = np.reshape(np.take(x, indices), (len(indices)))
		boundaries[:,1] = np.reshape(np.take(y, indices), (len(indices)))
		boundaries[:,2] = self.dt*self.__ml_dump_index*np.ones((boundaries.shape[0]))
		boundaries[:,3] = np.reshape(np.take(p, indices), (len(indices)))
		boundaries[:,4] = np.reshape(np.take(u, indices), (len(indices)))
		boundaries[:,5] = np.reshape(np.take(v, indices), (len(indices)))

		#Deletes boundary points from inner points
		pts = np.array([np.ndarray.flatten(np.delete(x, indices))])
		pts = np.append(pts, np.array([np.delete(y, indices)]), axis=0)
		pts = np.append(pts, np.array([np.delete(p, indices)]), axis=0)
		pts = np.append(pts, np.array([np.delete(u, indices)]), axis=0)
		pts = np.append(pts, np.array([np.delete(v, indices)]), axis=0)
		pts = np.insert(pts, 2, self._KernelAcoustic__ml_dump_index*np.ones((1,pts.shape[1])), axis=0)
		pts = pts.T #Make each variable a column (x,y,t,p,u,v)

		return pts, boundaries
	
	def dump(self):
		if (__ML_DUMP__):
			#fname = "data/wave/dump{:03d}.npz".format(self.__ml_dump_index)
			fname = self.dumps_dir + "/dump{:03d}.npz".format(self.__ml_dump_index)
			pts, boundaries = self.clean_data()
			f = open(fname, "w")
			np.savez(fname, pts = pts, bound = boundaries)
			f.close()
			self.__ml_dump_index += self.ml_dump_step

		# if __ML_DUMP__	
	# def dump	
# def KernelAcoustic
