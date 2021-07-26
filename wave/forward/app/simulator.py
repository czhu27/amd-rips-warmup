import numpy as np
import time
import os

import math

from mesh import Mesh
from field import Field
from integrator import IntegratorRK2, IntegratorRK3
from kernel import KernelAcoustic
import sys
import yaml
import datetime
sys.path.append(sys.path[0] + "/../..")
from data import process_wave_data_sample
from plots import make_heatmap_animation

# ------------------------------------------------------------------------------
# Simulator: main driver
# ------------------------------------------------------------------------------
class Simulator:
	def __init__(self, params):

		# Save ref to parameters
		self.__params = params

		timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		self.__params["data_dir"] = (self.__params["data_dir"] + 
			"/" + timestamp)

		# Create save dir
		os.makedirs(self.__params["data_dir"])

		# Create mesh and field
		tic = time.time()
		self.__mesh = Mesh(params)
		toc = time.time()
		self.__field = Field(params, self.__mesh)

		# Create containers for variables and right-hand sides
		self.__field.create_fields_acoustic()
		self.__field.initialize(self.__params["initializer"])
		
		# Create integrator and kernel
		if (self.__params["integrator"] == 'rk2'):
			self.__integrator = IntegratorRK2(self.__field)
		elif (self.__params["integrator"] == 'rk3'):	
			self.__integrator = IntegratorRK3(self.__field)
		else:
			print("   Invalid integrator!" + self.__params["integrator"])

		self.__kernel = KernelAcoustic(self.__mesh, self.__field, params)	
		print("Initialization time: " + "{:.2F} s".format(toc - tic))

	# def __init__

	def run(self):
		# Track elapsed time: 
		# 0: total
		# 1: volume
		# 2: fluxes
		# 3: wrapup
		# 4: integration
		tic = np.zeros(5)
		elapsed_time = np.zeros(5)
		tic[0] = time.time()
		
		# Dump initial conditions (if required -- test in dump() function)
		self.__kernel.dump()
			
		# Kernel: compute mass matrices (constant for all time steps)
		self.__kernel.compute_mass()
		
		# Time step until final time is reached and only if time 
		# step is large enough (a tiny time step may result from rounding, 
		# and there is no point going forward in time if by such a tiny amount)
		tf = self.__params["tf"]
		dt = self.__params["dt"]
		show_every = self.__params["show_every"]
		current_time = 0
		num_steps = math.ceil(self.__params["tf"]/self.__params["dt"])
		step_number = 0
		step_size = int(self.__params["sample_step"] / dt)
		#self.__field.display_p(step_number)
		while step_number < num_steps: # (current_time < tf) and (dt > 1e-15):
			
			# Initialize time integrator
			self.__integrator.initialize_step(current_time, dt)

			# Run integrator for current time step
			while self.__integrator.is_running():
				# Initialize stage
				self.__kernel.t = self.__integrator.t
				tic[4] = time.time()
				self.__integrator.initialize_stage()
				elapsed_time[4] += time.time() - tic[4]

				# Kernel: set all right-hand sides to 0!
				self.__kernel.reset()
				
				# Kernel: compute volume
				tic[1] = time.time()
				self.__kernel.compute_volume()
				elapsed_time[1] += time.time() - tic[1]

				# Kernel: compute fluxes
				tic[2] = time.time()
				self.__kernel.compute_fluxes()
				elapsed_time[2] += time.time() - tic[2]

				# Kernel: compute source
				self.__kernel.compute_sources(self.__integrator.t)

				# kernel: wrap (scale rhs by mass matrix)
				tic[3] = time.time()
				self.__kernel.wrapup()
				elapsed_time[3] += time.time() - tic[3]
				
				# Prep fields for next stage
				tic[4] = time.time()
				self.__integrator.next_stage()
				elapsed_time[4] += time.time() - tic[4]
			# while

			# Integrator's update step (adding up ki's and update field values)
			tic[4] = time.time()
			self.__integrator.update()
			elapsed_time[4] += time.time() - tic[4]
			
			# Increment time step and report
			current_time += dt	
			step_number += 1
			if (step_number%show_every == 0):
				print("[" + str(step_number) + "] " + 
				str(round(current_time, 3)))
			
			# Record trace (indexing starts at 0, so subtract 1 to step number) 
			self.__kernel.record_receivers(step_number - 1, current_time)

			# Make sure we stop exactly at final time 'tf'
			if (current_time + dt > tf):
				dt = tf - current_time
			
			# Simulation dump (flag in kernel.py must be True for 
			# simulation data to be saved into files!)
			if (step_number%step_size == 0):
				self.__kernel.dump()

			# Display pressure field at end of time step
			if (step_number%100 == 0):
				self.__field.display_p(step_number)

		# while	

		elapsed_time[0] = time.time() - tic[0]
		print("Total simulation time: " + "{:.2F} s".format(elapsed_time[0]))
		print("   Volume:             " + "{:.2F} s".format(elapsed_time[1]))
		print("   Fluxes:             " + "{:.2F} s".format(elapsed_time[2]))
		print("   Wrapup:             " + "{:.2F} s".format(elapsed_time[3]))
		print("   Integration:        " + "{:.2F} s".format(elapsed_time[4]))
		
		# Display pressure field at end of simulation
		self.__field.display_p(step_number)

	# def run

	def finalize(self):
		min_max = self.__field.display_diagnostics()
		self.__field.compute_errors(self.__params)
		self.__field.save_traces()

		# Process data
		process_wave_data_sample(self.__params["data_dir"], self.__params)
		yaml.safe_dump(self.__params, open(self.__params["data_dir"] + "/sim_configs.yaml", "w"))

		return min_max
	# def finalize

# class Simulator
