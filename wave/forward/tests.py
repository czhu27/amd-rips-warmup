import sys
sys.path.append(sys.path[0] + "/app")

import numpy as np
import functools
from mesh import Mesh
from element import Element2d
from field import Field
from quadrature import Gauss2d, GLL2d
from simulator import Simulator

tol_fp32 = 5e-7

def print_passed(test_name):
	print("\033[92mPASSED\033[0m: " + test_name)
# def print_passed

def print_failed(test_name):
	print("\033[91mFAILED\033[0m: " + test_name)
# def print_failed	

# ------------------------------------------------------------------------------
# Mesh test - Isomorphisms
# ------------------------------------------------------------------------------
test_name = "Mesh test - Isomorphisms (compute domain surface area)"

params = {
		"x0": 0.0,			# domain dimension (x0)
		"x1": 0.8,			# domain dimension (x1)
		"y0": 0.0,			# domain dimension (y0)
		"y1": 0.4,			# domain dimension (y1)
		"nx": 7,			# number of mesh elements (x)
		"ny": 3,			# number of mesh elements (y)
		"ps": 2,			# polynomial degree for state
		"pm": 0,			# polynomial degree for material
	}
mesh = Mesh(params)	

# Perform numerical integration, one point per element 
gauss2d = Gauss2d(1)
mesh.compute_isomorphisms(gauss2d.coordinates)
iso = mesh.isomorphisms

# Dot product is \sum_l J_l w_l per element.  Sum is summation over elements
mesh_area = np.sum(np.dot(iso[:,:,0], gauss2d.weights))

exact_area = (params["x1"] - params["x0"])*(params["y1"] - params["y0"])
test_ok = np.isclose(mesh_area, exact_area, rtol = tol_fp32)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
print("   Approx area: " + str(mesh_area))
print("   Exact area:  " + str(exact_area))

# ------------------------------------------------------------------------------
# Quadrature test - Gauss 2x2
# ------------------------------------------------------------------------------
test_name = "Quadrature test - Gauss 2x2"

params = {
		"x0": 0.0,			# domain dimension (x0)
		"x1": 1.0,			# domain dimension (x1)
		"y0": 0.0,			# domain dimension (y0)
		"y1": 1.0,			# domain dimension (y1)
		"nx": 4,			# number of mesh elements (x)
		"ny": 4,			# number of mesh elements (y)
		"ps": 2,			# polynomial degree for state
		"pm": 0,			# polynomial degree for material
	}
mesh = Mesh(params)	

# Perform numerical integration, one point per element 
gauss2d = Gauss2d(2)
mesh.compute_isomorphisms(gauss2d.coordinates)
iso = mesh.isomorphisms
global_coordinates = mesh.global_coordinates

# Evaluate function at all global coordinates
# f = a[0]x^2y^2 + a[1]x^2 + a[2]y^2 + a[3]xy + a[4]x + a[5]y + a[6]
a = np.random.rand(7)
vals = np.zeros((mesh.ne, gauss2d.num_points))
for k in range(gauss2d.num_points):
	x = global_coordinates[:, 2*k + 0]
	y = global_coordinates[:, 2*k + 1]
	vals[:, k] = a[0]*x**2*y**2 + \
				 a[1]*x**2 + \
				 a[2]*y**2 + \
				 a[3]*x*y + \
				 a[4]*x + \
				 a[5]*y + \
				 a[6]
# for k

# Dot product is \sum_l J_l w_l per element.  Sum is summation over elements
approx_integral = np.sum(np.dot(vals*iso[:,:,0], gauss2d.weights))
exact_integral = a[0]/9.0 + a[1]/3.0 + a[2]/3.0 + \
				 a[3]/4.0 + a[4]/2.0 + a[5]/2.0 + a[6]
test_ok = np.isclose(approx_integral, exact_integral, rtol = tol_fp32)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
print("   Approx: " + str(approx_integral))
print("   Exact : " + str(exact_integral))

# ------------------------------------------------------------------------------
# Quadrature test - Gauss 3x3
# ------------------------------------------------------------------------------
test_name = "Quadrature test - Gauss 3x3"

params = {
		"x0": 0.0,			# domain dimension (x0)
		"x1": 1.0,			# domain dimension (x1)
		"y0": 0.0,			# domain dimension (y0)
		"y1": 1.0,			# domain dimension (y1)
		"nx": 4,			# number of mesh elements (x)
		"ny": 4,			# number of mesh elements (y)
		"ps": 2,			# polynomial degree for state
		"pm": 0,			# polynomial degree for material
	}
mesh = Mesh(params)	

# Perform numerical integration, one point per element 
gauss2d = Gauss2d(3)
mesh.compute_isomorphisms(gauss2d.coordinates)
iso = mesh.isomorphisms
global_coordinates = mesh.global_coordinates

# Evaluate function at all global coordinates
# f = a[0]x^4 + a[1]y^4
a = np.random.rand(2)
vals = np.zeros((mesh.ne, gauss2d.num_points))
for k in range(gauss2d.num_points):
	x = global_coordinates[:, 2*k + 0]
	y = global_coordinates[:, 2*k + 1]
	vals[:, k] = a[0]*x**4 + a[1]*y**4 
# for k

# Dot product is \sum_l J_l w_l per element.  Sum is summation over elements
approx_integral = np.sum(np.dot(vals*iso[:,:,0], gauss2d.weights))
exact_integral = a[0]/5.0 + a[1]/5.0
test_ok = np.isclose(approx_integral, exact_integral, rtol = tol_fp32)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
print("   Approx: " + str(approx_integral))
print("   Exact : " + str(exact_integral))

# ------------------------------------------------------------------------------
# Quadrature test - GLL 2x2
# ------------------------------------------------------------------------------
test_name = "Quadrature test - GLL 2x2"

params = {
		"x0": 0.0,			# domain dimension (x0)
		"x1": 1.0,			# domain dimension (x1)
		"y0": 0.0,			# domain dimension (y0)
		"y1": 1.0,			# domain dimension (y1)
		"nx": 4,			# number of mesh elements (x)
		"ny": 4,			# number of mesh elements (y)
		"ps": 2,			# polynomial degree for state
		"pm": 0,			# polynomial degree for material
	}
mesh = Mesh(params)	

# Perform numerical integration, one point per element 
gll2d = GLL2d(2)
mesh.compute_isomorphisms(gll2d.coordinates)
iso = mesh.isomorphisms
global_coordinates = mesh.global_coordinates

# Evaluate function at all global coordinates
# f = a[0]x^2y^2 + a[1]x^2 + a[2]y^2 + a[3]xy + a[4]x + a[5]y + a[6]
a = np.random.rand(4)
vals = np.zeros((mesh.ne, gll2d.num_points))
for k in range(gll2d.num_points):
	x = global_coordinates[:, 2*k + 0]
	y = global_coordinates[:, 2*k + 1]
	vals[:, k] = a[0]*x*y + \
				 a[1]*x + \
				 a[2]*y + \
				 a[3]
# for k

# Dot product is \sum_l J_l w_l per element.  Sum is summation over elements
approx_integral = np.sum(np.dot(vals*iso[:,:,0], gll2d.weights))
exact_integral = a[0]/4.0 + a[1]/2.0 + a[2]/2.0 + a[3]
test_ok = np.isclose(approx_integral, exact_integral, rtol = tol_fp32)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
print("   Approx: " + str(approx_integral))
print("   Exact : " + str(exact_integral))

# ------------------------------------------------------------------------------
# Quadrature test - GLL 3x3
# ------------------------------------------------------------------------------
test_name = "Quadrature test - GLL 3x3"

params = {
		"x0": 0.0,			# domain dimension (x0)
		"x1": 1.0,			# domain dimension (x1)
		"y0": 0.0,			# domain dimension (y0)
		"y1": 1.0,			# domain dimension (y1)
		"nx": 4,			# number of mesh elements (x)
		"ny": 4,			# number of mesh elements (y)
		"ps": 2,			# polynomial degree for state
		"pm": 0,			# polynomial degree for material
	}
mesh = Mesh(params)	

# Perform numerical integration, one point per element 
gll2d = GLL2d(3)
mesh.compute_isomorphisms(gll2d.coordinates)
iso = mesh.isomorphisms
global_coordinates = mesh.global_coordinates

# Evaluate function at all global coordinates
# f = a[0]x^2y^2 + a[1]x^2 + a[2]y^2 + a[3]xy + a[4]x + a[5]y + a[6]
a = np.random.rand(7)
vals = np.zeros((mesh.ne, gll2d.num_points))
for k in range(gll2d.num_points):
	x = global_coordinates[:, 2*k + 0]
	y = global_coordinates[:, 2*k + 1]
	vals[:, k] = a[0]*x**2*y**2 + \
				 a[1]*x**2 + \
				 a[2]*y**2 + \
				 a[3]*x*y + \
				 a[4]*x + \
				 a[5]*y + \
				 a[6]
# for k

# Dot product is \sum_l J_l w_l per element.  Sum is summation over elements
approx_integral = np.sum(np.dot(vals*iso[:,:,0], gll2d.weights))
exact_integral = a[0]/9.0 + a[1]/3.0 + a[2]/3.0 + \
				 a[3]/4.0 + a[4]/2.0 + a[5]/2.0 + a[6]
test_ok = np.isclose(approx_integral, exact_integral, rtol = tol_fp32)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
print("   Approx: " + str(approx_integral))
print("   Exact : " + str(exact_integral))

# ------------------------------------------------------------------------------
# Element test p = 1
# ------------------------------------------------------------------------------
element = Element2d(1)
xi_1d = element.xi
num_points = len(xi_1d)
local_coordinates = np.zeros((num_points*num_points, 2), dtype = np.float32)
index = 0
for j in range(num_points):
	for i in range(num_points):
		local_coordinates[index, 0] = xi_1d[i]
		local_coordinates[index, 1] = xi_1d[j]
		index += 1
	# for i
# for j	
psi = element.evaluate_psi(local_coordinates)
test_ok = np.allclose(psi, np.identity(num_points*num_points), rtol = tol_fp32)
test_name = "Element2d test - Shape function values for p = 1"
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)

dpsi0, dpsi1 = element.evaluate_dpsi(local_coordinates)
test_ok = np.isclose(np.sum(dpsi0), 0.0, rtol = tol_fp32) and \
		  np.isclose(np.sum(dpsi1), 0.0, rtol = tol_fp32)
test_name = "Element2d test - Shape function derivatives for p = 1"
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
		  
# ------------------------------------------------------------------------------
# Element test p = 2
# ------------------------------------------------------------------------------
element = Element2d(2)
xi_1d = element.xi
num_points = len(xi_1d)
local_coordinates = np.zeros((num_points*num_points, 2), dtype = np.float32)
index = 0
for j in range(num_points):
	for i in range(num_points):
		local_coordinates[index, 0] = xi_1d[i]
		local_coordinates[index, 1] = xi_1d[j]
		index += 1
	# for i
# for j	
psi = element.evaluate_psi(local_coordinates)
test_ok = np.allclose(psi, np.identity(num_points*num_points), rtol = tol_fp32)
test_name = "Element2d test - Shape function values for p = 2"
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)

dpsi0, dpsi1 = element.evaluate_dpsi(local_coordinates)
test_ok = np.isclose(np.sum(dpsi0), 0.0, rtol = tol_fp32) and \
		  np.isclose(np.sum(dpsi1), 0.0, rtol = tol_fp32)
test_name = "Element2d test - Shape function derivatives for p = 2"
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
	
# ------------------------------------------------------------------------------
# Field test - Element global nodal coordinates
# ------------------------------------------------------------------------------
params = {
		"x0": 0.0,			# domain dimension (x0)
		"x1": 1.0,			# domain dimension (x1)
		"y0": 0.0,			# domain dimension (y0)
		"y1": 1.0,			# domain dimension (y1)
		"nx": 1,			# number of mesh elements (x)
		"ny": 1,			# number of mesh elements (y)
		"ps": 2,			# polynomial degree for state
		"pm": 0,			# polynomial degree for material
	}
mesh = Mesh(params)	
field = Field(params, mesh)
node_coordinates = field.global_node_coordinates[0]
node_coordinates_test = np.array([0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 
								  0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 
								  0.0, 1.0, 0.5, 1.0, 1.0, 1.0])
test_ok = np.allclose(node_coordinates, node_coordinates_test, rtol = tol_fp32)
test_name = "Field test - Element global node coordinates"
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)

# ------------------------------------------------------------------------------
# Interpolation helper functions
# ------------------------------------------------------------------------------
def interpolation_test_function(x, y, nx, ny):
	return np.sin(nx*np.pi*x)*np.sin(ny*np.pi*y)

def interpolation_test_dfunction(x, y, nx, ny):
	dfdx = nx*np.pi*np.cos(nx*np.pi*x)*np.sin(ny*np.pi*y)
	dfdy = np.sin(nx*np.pi*x)*ny*np.pi*np.cos(ny*np.pi*y)
	return dfdx, dfdy

def interpolation_error(params, fn):
	gauss = Gauss2d(params["int_diag"])

	# Create mesh and field
	mesh = Mesh(params)	
	field = Field(params, mesh)

	# Node coordinates are global coordinates of computational
	# nodes for the selected element (as defined in params)
	# node_coordinates[e] contains node coordinates for element e,
	# ordered as follows: x0 y0 x1 y0 ...
	node_coordinates = field.global_node_coordinates

	# Evaluate f(x,y) at nodal locations
	interp_vals = np.zeros((mesh.ne, field.num_nodes_per_element))
	for n in range(field.num_nodes_per_element):
		x = node_coordinates[:, 2*n + 0]
		y = node_coordinates[:, 2*n + 1]
		interp_vals[:, n] = fn(x, y)
	# for n

	# Create interpolative element.
	# Evaluate shape function values at integration points.  
	# Shape of psi is num_shape_functions x num_integration_points.
	element = Element2d(params["ps"])
	psi = element.evaluate_psi(gauss.coordinates)

	# Compute values at integration points for each element
	# Shape of vh is num_elements x num_integration_points
	vh = np.dot(interp_vals, psi)

	# Compute exact values at integration points
	mesh.compute_isomorphisms(gauss.coordinates)
	iso = mesh.isomorphisms
	global_coordinates = mesh.global_coordinates
	ve = np.zeros((mesh.ne, gauss.num_points))
	for n in range(gauss.num_points):
		x = global_coordinates[:, 2*n + 0]
		y = global_coordinates[:, 2*n + 1]
		ve[:, n] = fn(x, y)
	# for n

	# Numerical integration of L2 norm of error
	diff_squared = (vh - ve)**2		# function to integrate
	err = np.sqrt(np.sum(np.dot(diff_squared*iso[:,:,0], gauss.weights)))

	return err

# def interpolation_error

def interpolation_derivative_error(params, fn, dfn):
	gauss = Gauss2d(params["int_diag"])

	# Create mesh and field
	mesh = Mesh(params)	
	field = Field(params, mesh)

	# Node coordinates are global coordinates of computational
	# nodes for the selected element (as defined in params)
	# node_coordinates[e] contains node coordinates for element e,
	# ordered as follows: x0 y0 x1 y0 ...
	node_coordinates = field.global_node_coordinates

	# Evaluate f(x,y) at nodal locations
	interp_vals = np.zeros((mesh.ne, field.num_nodes_per_element))
	for n in range(field.num_nodes_per_element):
		x = node_coordinates[:, 2*n + 0]
		y = node_coordinates[:, 2*n + 1]
		interp_vals[:, n] = fn(x, y)
	# for n

	# Create interpolative element.
	# Evaluate shape function derivatives at integration points.  
	# Shape of dpsi0, dpsi1 is num_shape_functions x num_integration_points.
	element = Element2d(params["ps"])
	dpsi0, dpsi1 = element.evaluate_dpsi(gauss.coordinates) # dpsi/dxi, dpsi/deta

	# Compute isomorphisms at integration points, and global coordinates
	mesh.compute_isomorphisms(gauss.coordinates)
	iso = mesh.isomorphisms
	
	# Compute values at integration points for each element
	# Shape of dvhdx and dvhdy is (num_elements, num_integration_points)
	dvhdx = np.dot(interp_vals, dpsi0)*iso[:,:,1] + \
			np.dot(interp_vals, dpsi1)*iso[:,:,3]
	dvhdy = np.dot(interp_vals, dpsi0)*iso[:,:,2] + \
			np.dot(interp_vals, dpsi1)*iso[:,:,4]

	# Compute exact values at integration points
	global_coordinates = mesh.global_coordinates
	dvedx = np.zeros((mesh.ne, gauss.num_points))
	dvedy = np.zeros((mesh.ne, gauss.num_points))
	for n in range(gauss.num_points):
		x = global_coordinates[:, 2*n + 0]
		y = global_coordinates[:, 2*n + 1]
		dvedx[:, n], dvedy[:, n] = dfn(x, y)
	# for n

	# Numerical integration of L2 norm of error
	diff_squared = (dvhdx - dvedx)**2 + (dvhdy - dvedy)**2	
	err = np.sqrt(np.sum(np.dot(diff_squared*iso[:,:,0], gauss.weights)))

	return err

# def interpolation_derivative_error

# ------------------------------------------------------------------------------
# Interpolation test - Element p = 1
# ------------------------------------------------------------------------------
test_name = "Interpolation test - Element p = 1"

# interpolation error 1
params1 = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,		
		   "nx": 8, "ny": 8,
		   "ps": 1, "pm": 0, "int_diag": 3}
fn = functools.partial(interpolation_test_function, nx = 1, ny = 1)		   
err1 = interpolation_error(params1, fn)

# Interpolation error 2
params2 = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,		
		   "nx": 16, "ny": 16,
		   "ps": 1, "pm": 0, "int_diag": 3}
err2 = interpolation_error(params2, fn)
test_ok = np.isclose(err1/err2, 4.0, rtol = 0.01)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
print("   Error 1: " + str(err1))
print("   Error 2: " + str(err2))
print("   Ratio:   " + str(err1/err2))

# ------------------------------------------------------------------------------
# Interpolation derivative test - Element p = 1
# ------------------------------------------------------------------------------
test_name = "Interpolation derivative test - Element p = 1"

# interpolation error 1
params1 = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,		
		   "nx": 8, "ny": 8,
		   "ps": 1, "pm": 0, "int_diag": 3}
fn = functools.partial(interpolation_test_function, nx = 1, ny = 1)		   
dfn = functools.partial(interpolation_test_dfunction, nx = 1, ny = 1)	
err1 = interpolation_derivative_error(params1, fn, dfn)

# Interpolation error 2
params2 = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,		
		   "nx": 16, "ny": 16,
		   "ps": 1, "pm": 0, "int_diag": 3}
err2 = interpolation_derivative_error(params2, fn, dfn)
test_ok = np.isclose(err1/err2, 2.0, rtol = 0.01)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
print("   Error 1: " + str(err1))
print("   Error 2: " + str(err2))
print("   Ratio:   " + str(err1/err2))

# ------------------------------------------------------------------------------
# Interpolation test - Element p = 2
# ------------------------------------------------------------------------------
test_name = "Interpolation test - Element p = 2"
fn = functools.partial(interpolation_test_function, nx = 2, ny = 2)		   

# interpolation error 1
params1 = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,		
		   "nx": 8, "ny": 8,
		   "ps": 2, "pm": 0, "int_diag": 3}
err1 = interpolation_error(params1, fn)

# Interpolation error 2
params2 = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,		
		   "nx": 16, "ny": 16,
		   "ps": 2, "pm": 0, "int_diag": 3}
err2 = interpolation_error(params2, fn)
test_ok = np.isclose(err1/err2, 8.0, rtol = 0.01)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
print("   Error 1: " + str(err1))
print("   Error 2: " + str(err2))
print("   Ratio:   " + str(err1/err2))

# ------------------------------------------------------------------------------
# Interpolation derivative test - Element p = 2
# ------------------------------------------------------------------------------
test_name = "Interpolation derivative test - Element p = 2"

# interpolation error 1
params1 = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,		
		   "nx": 8, "ny": 8,
		   "ps": 2, "pm": 0, "int_diag": 3}
fn = functools.partial(interpolation_test_function, nx = 2, ny = 2)		   
dfn = functools.partial(interpolation_test_dfunction, nx = 2, ny = 2)	
err1 = interpolation_derivative_error(params1, fn, dfn)

# Interpolation error 2
params2 = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,		
		   "nx": 16, "ny": 16,
		   "ps": 2, "pm": 0, "int_diag": 3}
err2 = interpolation_derivative_error(params2, fn, dfn)
test_ok = np.isclose(err1/err2, 4.0, rtol = 0.01)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)
print("   Error 1: " + str(err1))
print("   Error 2: " + str(err2))
print("   Ratio:   " + str(err1/err2))

# ------------------------------------------------------------------------------
# Full simulation - p = 1, RK2
# ------------------------------------------------------------------------------
test_name = "Simulation test - p1, rk2"

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
	"tf": 0.2,			# final time
	"dt": 0.01,			# time step
	"show_every": 10,	# interval between two time steps reports
	"integrator": 'rk2',# time integrator 
	# Discretization-related parameters
	"ps": 1,			# polynomial degree for state variables
	"pm": 0,			# polynomial degree for material properties	
	"int_diag": 3,		# integration used for diagnostics (num 1D Gauss)  
	# Initialization function
	"initializer": 'gaussian'
}	

# Create simulator and run
simulator = Simulator(params)
simulator.run()
min_max = simulator.finalize()
min_max_expected = np.array([[-0.10407929, 0.25662514],
							 [-0.34743553, 0.34743553],
							 [-0.34743553, 0.34743553]])
test_ok = np.allclose(min_max, min_max_expected, rtol=tol_fp32)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)

# ------------------------------------------------------------------------------
# Full simulation - p = 2, RK3
# ------------------------------------------------------------------------------
test_name = "Simulation test - p2, rk3"

# Create all parameters
params = {
	# Space-related parameters
	"x0": 0.0,			# domain dimension (x0)
	"x1": 1.0,			# domain dimension (x1)
	"y0": 0.0,			# domain dimension (y0)
	"y1": 1.0,			# domain dimension (y1)
	"nx": 8,			# number of mesh elements (x)
	"ny": 8,			# number of mesh elements (y)
	# Time-related parameters
	"tf": 0.2,			# final time
	"dt": 0.01,			# time step
	"show_every": 10,	# interval between two time steps reports
	"integrator": 'rk3',# time integrator 
	# Discretization-related parameters
	"ps": 2,			# polynomial degree for state variables
	"pm": 0,			# polynomial degree for material properties	
	"int_diag": 3,		# integration used for diagnostics (num 1D Gauss)  
	# Initialization function
	"initializer": 'gaussian'
}	

# Create simulator and run
simulator = Simulator(params)
simulator.run()
min_max = simulator.finalize()
min_max_expected = np.array([[-0.17617202, 0.26225159],
							 [-0.33359649, 0.33359649],
							 [-0.33359649, 0.33359649]])
test_ok = np.allclose(min_max, min_max_expected, rtol=tol_fp32)
if (test_ok):
	print_passed(test_name)
else:
	print_failed(test_name)

