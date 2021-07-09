import numpy as np

# ------------------------------------------------------------------------------
# class IntegratorRK2
# ------------------------------------------------------------------------------
class IntegratorRK2:
	def __init__(self, field):
		self.field = field
		
		# Parameters specific to this integrator
		self.__num_stages = 2
		self.__current_stage = 0
		self.__c = np.array([0.0, 0.5], dtype = np.float32)
		self.__b = np.array([0.0, 1.0], dtype = np.float32)
		self.__a = 0.5

		# Will hold current time info
		self.__t0 = 0.0	# time at beginning of step
		self.t = 0.0	# current time (during course of integration)
		self.__dt = 0.0	# current time step

		# Create k1 and k2 (same shape as fields)
		self.old = {}	# to store copy of field at previous step
		self.k1 = {}
		self.k2 = {}
		for f in field.val:
			self.old[f] = np.zeros(field.val[f].shape, dtype = np.float32)
			self.k1[f] = np.zeros(field.val[f].shape, dtype = np.float32)
			self.k2[f] = np.zeros(field.val[f].shape, dtype = np.float32)
	# def __init__
	
	# Initialize step: save time as of beginning of step and
	# make a copy of field values
	def initialize_step(self, t, dt):
		self.__current_stage = 0
		self.__t0 = t
		self.__dt = dt
		# Save old field values (makes an *actual* copy --> modifying
		# self.old will not change field.val!)
		for f in self.old:
			self.old[f] = np.copy(self.field.val[f])
	
	# Inquire integrator if it's still running
	def is_running(self):
		return (self.__current_stage < self.__num_stages)
	# def is_running	

	# Initialize stage
	def initialize_stage(self):
		self.t = self.__t0 + self.__c[self.__current_stage]*self.__dt
	# def initialize_step	

	# Prep fields and time for next stage
	def next_stage(self):
		
		# If done with first stage, save rhs values into k1
		# and update field values for next stage's rhs computation
		if (self.__current_stage == 0):
			for f in self.k1:
				self.k1[f] = np.copy(self.field.rhs[f])
				self.field.val[f] = self.old[f] + self.__a*self.__dt*self.k1[f]
		
		# If done with second stage, save rhs values into k2
		if (self.__current_stage == 1):
			for f in self.k2:
				self.k2[f] = np.copy(self.field.rhs[f])

		self.__current_stage += 1

	# Obtain new field values based on k1 and k2
	def update(self):
		for f in self.field.val:
			self.field.val[f] = np.copy(self.old[f] + 
										self.__dt*(self.__b[0]*self.k1[f] + 
												   self.__b[1]*self.k2[f]))
		

# def class IntegratorRK2

# ------------------------------------------------------------------------------
# class IntegratorRK3
# ------------------------------------------------------------------------------
class IntegratorRK3:
	def __init__(self, field):
		self.field = field
		
		# Parameters specific to this integrator
		self.__num_stages = 3
		self.__current_stage = 0
		self.__c = np.array([0.0, 0.5, 1.0], dtype = np.float32)
		self.__b = np.array([1./6., 2./3., 1./6.], dtype = np.float32)
		self.__a21 = 0.5
		self.__a31 = -1.0
		self.__a32 = 2.0

		# Will hold current time info
		self.__t0 = 0.0	# time at beginning of step
		self.t = 0.0	# current time (during course of integration)
		self.__dt = 0.0	# current time step

		# Create k1 and k2 (same shape as fields)
		self.old = {}	# to store copy of field at previous step
		self.k1 = {}
		self.k2 = {}
		self.k3 = {}
		for f in field.val:
			self.old[f] = np.zeros(field.val[f].shape, dtype = np.float32)
			self.k1[f] = np.zeros(field.val[f].shape, dtype = np.float32)
			self.k2[f] = np.zeros(field.val[f].shape, dtype = np.float32)
			self.k3[f] = np.zeros(field.val[f].shape, dtype = np.float32)
	# def __init__
	
	# Initialize step: save time as of beginning of step and
	# make a copy of field values
	def initialize_step(self, t, dt):
		self.__current_stage = 0
		self.__t0 = t
		self.__dt = dt
		# Save old field values (makes an *actual* copy --> modifying
		# self.old will not change field.val!)
		for f in self.old:
			self.old[f] = np.copy(self.field.val[f])
	
	# Inquire integrator if it's still running
	def is_running(self):
		return (self.__current_stage < self.__num_stages)
	# def is_running	

	# Initialize stage
	def initialize_stage(self):
		self.t = self.__t0 + self.__c[self.__current_stage]*self.__dt
	# def initialize_stage	

	def get_stage_number(self):
		return self.__current_stage;

	# Prep fields and time for next stage
	def next_stage(self):
		
		# If done with first stage, save rhs values into k1
		# and update field values for next stage's rhs computation
		if (self.__current_stage == 0):
			for f in self.k1:
				self.k1[f] = np.copy(self.field.rhs[f])
				self.field.val[f] = self.old[f] + \
								 	self.__a21*self.__dt*self.k1[f]
		
		# If done with second stage, save rhs values into k2
		# and update field values for next stage's rhs computation
		if (self.__current_stage == 1):
			for f in self.k2:
				self.k2[f] = np.copy(self.field.rhs[f])
				self.field.val[f] = self.old[f] + \
									self.__a31*self.__dt*self.k1[f] + \
									self.__a32*self.__dt*self.k2[f]

		# If done with third stage, save rhs
		if (self.__current_stage == 2):
			for f in self.k3:
				self.k3[f] = np.copy(self.field.rhs[f])
			
		self.__current_stage += 1

	# Obtain new field values based on k1, k2, and k3
	def update(self):
		for f in self.field.val:
			self.field.val[f] = np.copy(self.old[f] + 
										self.__dt*(self.__b[0]*self.k1[f] + 
												   self.__b[1]*self.k2[f] +
												   self.__b[2]*self.k3[f]))
		

# def class IntegratorRK3
