import numpy as np

def data_creation(params, corners):
	N_f_int = params[0]
	N_f_ext = params[1]
	N_f_border = params[2]
	N_f_intl = int(N_f_int * params[3])
	N_f_extl = int(N_f_ext * params[4])
	N_f_intul = int(N_f_int * (1 - params[3]))
	N_f_extul = int(N_f_ext * (1 - params[4]))
	X_f_int = np.zeros((N_f_int,2), dtype = np.float32)
	X_f_ext = np.zeros((N_f_ext,2), dtype = np.float32)
	X_f_border = np.zeros((N_f_border,2), dtype = np.float32)
	#Interior Points on [-1,1]
	X_f_int[:,0] = np.random.uniform(-1, 1, N_f_int)
	X_f_int[:,1] = np.random.uniform(-1, 1, N_f_int)
	#Exterior Points [-2,-1]U[1,2]
	def bad_zone(v):
		if -1 <= v[0] <= 1 and -1 <= v[1] <= 1:
			return True
		else:
			return False

	def generate_point():
		v = np.zeros((2,))
		while bad_zone(v):
			v = np.random.uniform(-2,2,(2,))
		v = v[None,:]
		return v

	for i in range(N_f_ext):
		temp = generate_point()
		X_f_ext[i,:] = temp
		
	#Border Points on box with (|x|,|y|) = (1,1)
	X_f_borderleft = np.array((-1*np.ones((N_f_border//4)), np.random.rand(N_f_border//4))).T
	X_f_borderright = np.array((np.ones((N_f_border//4)), np.random.rand(N_f_border//4))).T
	X_f_borderup = np.array((np.random.rand(N_f_border//4), np.ones((N_f_border//4)))).T
	X_f_borderdown = np.array((np.random.rand(N_f_border//4), -1*np.ones((N_f_border//4)))).T
	X_f_bordermore = np.array((np.random.rand(N_f_border%4), -1*np.ones((N_f_border%4)))).T
	X_f_border[:,0] = np.concatenate((X_f_borderleft[:,0], X_f_borderright[:,0],
                            X_f_borderup[:,0], X_f_borderdown[:,0], X_f_bordermore[:,0]))
	X_f_border[:,1] = np.concatenate((X_f_borderleft[:,1], X_f_borderright[:,1],
                            X_f_borderup[:,1], X_f_borderdown[:,1],X_f_bordermore[:,1]))

	#Labeling Data
	X_f_l = np.zeros((N_f_intl + N_f_border + N_f_extl,2), dtype = np.float32)
	X_f_ul = np.zeros((N_f_intul + N_f_extul,2), dtype = np.float32)
	X_f_l[0:N_f_intl] = X_f_int[0:N_f_intl]
	X_f_l[N_f_intl:N_f_intl + N_f_border] = X_f_border[:,:]
	X_f_l[N_f_intl + N_f_border:] = X_f_ext[0:N_f_extl,:]
	X_f_ul[0:N_f_intul] = X_f_int[N_f_intl:]
	X_f_ul[N_f_intul:] = X_f_ext[N_f_extl:]

	#Add corners
	if corners:
		X_f_l[0, 0] = -2.0; X_f_l[0, 1] = -2.0
		X_f_l[1, 0] =  2.0; X_f_l[1, 1] = -2.0
		X_f_l[2, 0] =  2.0; X_f_l[2, 1] =  2.0
		X_f_l[3, 0] = -2.0; X_f_l[3, 1] =  2.0

	return X_f_l, X_f_ul

def create_meshgrid(lb, ub, step_size=0.01):
	'''
	Create a meshgrid on the square [lb, ub] with ((ub-lb)/step_size + 1)^2 points
	'''
	x0 = np.arange(lb, ub+step_size, step_size)
	x1 = np.arange(lb, ub+step_size, step_size)
	return np.meshgrid(x0, x1), x0.size

def compute_error(model, f, lb, ub):
	'''
	Compute L2-error of model against f, on the square [lb, ub]
	'''
	mesh, n1d = create_meshgrid(lb, ub)
	x0_g, x1_g = mesh
	npts = n1d*n1d

	f_true = f(x0_g, x1_g)

	ml_input = np.zeros((npts, 2))
	ml_input[:,0] = x0_g.flatten()
	ml_input[:,1] = x1_g.flatten()
	ml_output = model.predict(ml_input)
	
	f_ml = np.reshape(ml_output, (n1d, n1d), order = 'C')
	
	error = np.sqrt(np.mean(np.square(f_ml - f_true)))
	return error

def extrap_error(model, f, i_lb, i_ub, o_lb, o_ub, step_size=0.01):
	'''
	i_lb: lower inner bound, i_ub: upper inner bound
	o_lb: lower outer bound, o_ub: upper outer bound
	Compute L2-error of model against f, on the square [o_lb, o_ub], excluding the
	points in the square [i_lb, i_ub]
	'''
	mesh, n1d = create_meshgrid(o_lb, o_ub, step_size)
	x, y = mesh
	npts = n1d*n1d
	less_points = int((i_ub-i_lb)/step_size)+1
	npts = npts - less_points**2
	
	is_interior = ((x >= i_lb) & (x <= i_ub+step_size)) & ((y >= i_lb) & (y <= i_ub+step_size))
	x_ext = x[~is_interior]
	y_ext = y[~is_interior]

	f_true = f(x_ext, y_ext)

	ml_input = np.zeros((npts, 2))
	ml_input[:,0] = x_ext.flatten()
	ml_input[:,1] = y_ext.flatten()
	ml_output = model.predict(ml_input)

	f_ml = np.reshape(ml_output, (npts), order = 'C')

	error = np.sqrt(np.mean(np.square(f_ml - f_true)))
	return error