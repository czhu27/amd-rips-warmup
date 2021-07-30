import sys
import datetime
sys.path.append(sys.path[0] + "/../..")
from data import process_wave_data
sys.path.append(sys.path[0] + "/app")
from simulator import Simulator

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = "data/wave/" + timestamp
#data_dir = "data/wave/triple"

# Create all parameters
params = {
	# Space-related parameters
	"x0": 0.0,			# domain dimension (x0)
	"x1": 1.0,			# domain dimension (x1)
	"y0": 0.0,			# domain dimension (y0)
	"y1": 1.0,			# domain dimension (y1)
	"nx": 25,			# number of mesh elements (x)
	"ny": 25,			# number of mesh elements (y)
	# Time-related parameters
	"tf": 2.0,			# final time
	"dt": 0.002,		# time step
	"sample_step": .01,  # amount of time between samples, should be multiple of dt
	"show_every": 50,	# interval between two time steps reports
	"integrator": 'rk2',# time integrator 
	# Discretization-related parameters
	"ps": 2,			# polynomial degree for state variables
	"pm": 0,			# polynomial degree for material properties	
	"int_diag": 3,		# integration used for diagnostics (num 1D Gauss)  
	# Initialization function
	"initializer": 'gaussian',
	# Source characteristics (locations given as [[x0, y0], [x1, y1], ...)
	#"src_loc": [[0.25, 0.25], [0.75, 0.75]],
	"src_loc": [ [0.9, 0.1] ],
	"src_f": 5.0,		# source frequency (Hz)
	"src_a": 0.0,		# source magnitude (set to 0 to disable source)
	#"src_gauss": [[0.5, 0.5], [0.7, 0.1], [0.4, 0.8]], (default is [[0.5, 0.5]])
	"src_gauss": [[0.9, 0.1]],
	# Receivers coordinates (x0, y0, x1, y1, ...) 
	"rcv_loc": [ [0.25, 0.25], [0.60, 0.50] ],
	# Where to save this run
	"data_dir": data_dir,
	# Save percentages of [[interior, int bound, labeled], [exterior,ext bound, labeled], [int_test, ext_test]]
	"data_percents": [[.01, .01, 1.0, 1.0], [.01, .01, 0.0, 0.0], [.05, .05]],
	"heatmap": False,
	"data_plot": True,
	"seed": 0, 
	"int_ext_time": 1
}	

# Create simulator and run
simulator = Simulator(params)
simulator.run()
simulator.finalize()

# Process data here
# process_wave_data(data_dir, params)