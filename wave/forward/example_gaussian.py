import sys
import datetime
sys.path.append(sys.path[0] + "/app")
sys.path.append(sys.path[0] + "/../..")
from data import process_wave_data

from simulator import Simulator

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = "data/wave/" + timestamp

# Create all parameters
params = {
	# Space-related parameters
	"x0": 0.0,			# domain dimension (x0)
	"x1": 1.0,			# domain dimension (x1)
	"y0": 0.0,			# domain dimension (y0)
	"y1": 1.0,			# domain dimension (y1)
	"nx": 16,			# number of mesh elements (x)
	"ny": 16,			# number of mesh elements (y)
	# Time-related parameters
	"tf": 1.0,			# final time
	"dt": 0.01,			# time step
	"sample_step": 0.04,# amount of time between samples, should be multiple of dt
	"show_every": 10,	# interval between two time steps reports
	"integrator": 'rk2',# time integrator 
	# Discretization-related parameters
	"ps": 1,			# polynomial degree for state variables
	"pm": 0,			# polynomial degree for material properties	
	"int_diag": 3,		# integration used for diagnostics (num 1D Gauss)  
	# Initialization function
	"initializer": 'gaussian',
	"data_dir": "data/wave/" + timestamp
}	

# Create simulator and run
simulator = Simulator(params)
simulator.run()
simulator.finalize()

process_wave_data(data_dir, params)
