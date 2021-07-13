import sys
import datetime
sys.path.append(sys.path[0] + "/../..")
from data import process_wave_data
sys.path.append(sys.path[0] + "/app")
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
	"nx": 10,			# number of mesh elements (x)
	"ny": 10,			# number of mesh elements (y)
	# Time-related parameters
	"tf": 1,			# final time
	"dt": 0.002,		# time step
	#"sample_step":     # amount of time between samples, should be multiple of dt
	"show_every": 50,	# interval between two time steps reports
	"integrator": 'rk2',# time integrator 
	# Discretization-related parameters
	"ps": 2,			# polynomial degree for state variables
	"pm": 0,			# polynomial degree for material properties	
	"int_diag": 3,		# integration used for diagnostics (num 1D Gauss)  
	# Initialization function
	"initializer": 'zero',
	# Source characteristics (locations given as [[x0, y0], [x1, y1], ...)
	#"src_loc": [[0.25, 0.25], [0.75, 0.75]],
	"src_loc": [ [0.5, 0.5] ],
	"src_f": 5.0,		# source frequency (Hz)
	"src_a": 1.0,		# source magnitude (set to 0 to disable source)
	# Receivers coordinates (x0, y0, x1, y1, ...) 
	"rcv_loc": [ [0.25, 0.25], [0.60, 0.50] ],
	# Where to save this run
	"data_dir": data_dir
}	

# Create simulator and run
simulator = Simulator(params)
simulator.run()
simulator.finalize()

# Process data here
# process_wave_data(data_dir, params)