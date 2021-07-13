import numpy as np
import glob
import time

from helpers import unstack

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

def data_wave(params, time_steps = None, nx = None, ny = None, order = None):
	'''
	Data creation/reading for wave
	'''
	# Load the proper values
	if time_steps == None or nx == None or ny == None or order == None:
		print("Getting nx, ny, nodes from loaded data")
		fname = "data/wave/dump{:03d}.npz".format(0)
		f = open(fname, "r")
		loaded = np.load(fname, allow_pickle=True)
		points, nodes = loaded['p'].shape
		print("WARNING: Assuming input region is square")
		nx = ny = points ** 0.5
		assert abs(nx - int(nx)) < 0.001
		nx, ny = int(nx), int(ny)
		order = (nodes**0.5 - 1)
		assert abs(order - int(order)) < 0.001, "Reading invalid data, nodes isn't square."
		order = int(order)
		num_dumps = len(glob.glob("data/wave/dump*.npz"))
		print("WARNING: UNKNOWN TIME STEP")
		dt = 0.01
		time_steps = int(1 / dt)

	#Initialize vectors/constants
	num_elements = nx*ny
	nodes = (order+1)**2
	length = 2*time_steps*num_elements*nodes
	p = np.zeros((2*time_steps,num_elements,nodes), dtype=np.float32)
	u = np.zeros((2*time_steps,num_elements,nodes), dtype=np.float32)
	v = np.zeros((2*time_steps,num_elements,nodes), dtype=np.float32)
	xy = np.zeros((2*time_steps,num_elements,nodes*2), dtype=np.float32)
	x = np.zeros((2*time_steps,num_elements,nodes), dtype=np.float32)
	y = np.zeros((2*time_steps,num_elements,nodes), dtype=np.float32)

	#Read in data for wave equation up to t=2, t \in (1,2] for error calcs, etc.
	for i in range(time_steps):
		fname = "data/wave/dump{:03d}.npz".format(i)
		f = open(fname, "r")
		loaded = np.load(fname, allow_pickle=True)
		p[i,:,:] = loaded['p']
		u[i,:,:] = loaded['u']
		v[i,:,:] = loaded['v']
		xy[i,:,:] = loaded['xy']
		f.close()

	#Reshapes data into column vectors
	x = xy[:,:,0:18:2]
	y = xy[:,:,1:18:2]
	x_flat = np.reshape(x, (length,1))
	y_flat = np.reshape(y, (length,1))
	u_flat = np.reshape(u, (length,1))
	v_flat = np.reshape(v, (length,1))
	p_flat = np.reshape(p, (length,1))
	t_flat = np.zeros((length,1), dtype=np.float32)
	for i in range(time_steps):
		t_flat[i*nodes*num_elements:(i+1)*nodes*num_elements,0] = np.reshape(.02*i*np.ones(nodes*num_elements), (nodes*num_elements))

	#Label/Unlabeled Data
	N_w_intl = int(params[0]*params[3])
	N_w_borl = int(params[1]*params[4])
	N_w_extl = int(params[2]*params[5])
	N_w_intul = int(params[0]*(1 - params[3]))
	N_w_borul = int(params[1]*(1 - params[4]))
	N_w_extul = int(params[2]*(1 - params[5]))
	X_w_l = np.zeros((N_w_intl+N_w_borl+N_w_extl, 3), dtype=np.float32)
	Y_l = np.zeros((N_w_intl+N_w_borl+N_w_extl, 1), dtype=np.float32)
	X_w_ul = np.zeros((N_w_intul+N_w_borul+N_w_extul, 3), dtype=np.float32)
	
	#Makes labeled data vector
	#Pulls labeled data randomly from interior for domain t \in [0,1]
	for i in range(N_w_intl):
		rand = np.random.randint(0,length/2)
		X_w_l[i,:] = np.concatenate((x_flat[rand],y_flat[rand],t_flat[rand]))
		Y_l[i,:] = p_flat[rand]

	#Pulls labeled data from random border points in simulated range
	border_pointsleft = np.argwhere(x_flat == 0)
	border_pointsright = np.argwhere(x_flat == 1)
	border_pointsdown = np.argwhere(y_flat == 0)
	border_pointsup = np.argwhere(y_flat == 1)

	for i in range(N_w_borl):
		rand1 = np.random.rand(2)
		rand2 = np.random.randint(0,1000)
		if rand1[0] > 0.5:
			if rand1[1] > 0.5:
				X_w_l[N_w_intl+i] = np.concatenate((x_flat[border_pointsup[rand2][0]],y_flat[border_pointsup[rand2][0]], t_flat[border_pointsright[rand2][0]]))
				Y_l[N_w_intl+i] = p_flat[border_pointsup[rand2][0]]
			else:
				X_w_l[N_w_intl+i] = np.concatenate((x_flat[border_pointsright[rand2][0]],y_flat[border_pointsright[rand2][0]], t_flat[border_pointsright[rand2][0]]))
				Y_l[N_w_intl+i] = p_flat[border_pointsright[rand2][0]]
		else:
			if rand1[1] > 0.5:
				X_w_l[N_w_intl+i] = np.concatenate((x_flat[border_pointsdown[rand2][0]],y_flat[border_pointsdown[rand2][0]], t_flat[border_pointsright[rand2][0]]))
				Y_l[N_w_intl+i] = p_flat[border_pointsdown[rand2][0]]
			else:
				X_w_l[N_w_intl+i] = np.concatenate((x_flat[border_pointsleft[rand2][0]],y_flat[border_pointsleft[rand2][0]], t_flat[border_pointsright[rand2][0]]))
				Y_l[N_w_intl+i] = p_flat[border_pointsleft[rand2][0]]

	#Pulls labeled data randomly from exterior for domain t \in (1,2]
	for i in range(N_w_extl):
		rand = np.random.randint(0,len(X_w_l))
		X_w_l[i,:] = np.concatenate((x_flat[rand],y_flat[rand],t_flat[rand]))
	
	#Makes unlabeled data vector
	#Pulls unlabeled data randomly from interior for domain t \in (1,2]
	for i in range(N_w_intul):
		X_w_ul[i,:] = np.random.rand(3)

	#Makes unlabeled data for border
	for i in range(N_w_intul, N_w_intul + N_w_borul):
		rand1 = np.random.rand(2)
		if rand1[0] > 0.5:
			if rand1[1] > 0.5:
				X_w_ul[i] = np.array((rand1[0],1, np.random.randint(time_steps, 2*time_steps)))
			else:
				X_w_ul[i] = np.array((1,rand1[1], np.random.randint(time_steps, 2*time_steps)))
		else:
			if rand1[1] > 0.5:
				X_w_ul[i] = np.array((rand1[0],0, np.random.randint(time_steps, 2*time_steps)))
			else:
				X_w_ul[i] = np.array((0,rand1[1], np.random.randint(time_steps, 2*time_steps)))

	#Pulls unlabeled data randomly from exterior for domain t \in (1,2]
	for i in range(N_w_intul + N_w_borl, N_w_intul + N_w_borul + N_w_extul):
		X_w_ul[i,:] = np.concatenate((np.random.rand(2),np.random.randint(time_steps,2*time_steps, (1))))	

	return X_w_l, X_w_ul, Y_l, x_flat, y_flat, t_flat, p_flat



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

	ml_input = np.zeros((npts, 3))
	ml_input[:,0] = x0_g.flatten()
	ml_input[:,1] = x1_g.flatten()
	ml_input[:,2] = t*np.ones((npts))
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

def compute_error_wave(model, x, y, t, p):
	f_true = p
	ml_input = np.concatenate((x,y,t))
	ml_output = model.predict(ml_input)
	f_ml = np.reshape(ml_output, (len(p), 1))
	error = np.sqrt(np.mean(np.square(f_ml - f_true)))
	return error

def load_data(dump_file):
    '''
    Loads data from dump file
    '''
    stuff = np.load(dump_file)
    u = stuff['u']
    v = stuff['v']
    p = stuff['p']
    x, y = stuff['xy'][:,0:18:2], stuff['xy'][:,1:18:2]
    x_n,y_n,p_n,u_n,v_n = x.flatten(), y.flatten(), p.flatten(), u.flatten(), v.flatten()
    return x_n,y_n,p_n,u_n,v_n

def process_wave_data(wave_data_dir):
    tic = time.time()

    # TODO: This is bad
    tf = 5
    dt = 0.002
    T = int(tf / dt) + 1

    x,y,p,u,v = load_data(wave_data_dir + "/dumps/dump000.npz")
    slice_size = len(x)

    x_all, y_all, p_all, u_all, v_all = [np.zeros((T, slice_size)) for i in range(5)]
    t_all = np.zeros(T)

    for i in range(T): 
        dump_file = wave_data_dir + "/dumps/dump{:03d}.npz".format(i)
        x,y,p,u,v = load_data(dump_file)
        x_all[i, :] = x
        y_all[i, :] = y
        p_all[i, :] = p
        u_all[i, :] = u
        v_all[i, :] = v
        t_all[i] = dt * i

	# sample

    x = x_all.flatten()
    y = y_all.flatten()
    p = p_all.flatten()
    u = u_all.flatten()
    v = v_all.flatten()
    t = np.repeat(t_all, slice_size)

    data = np.stack([x,y,t,p,u,v], axis=-1)

    # Reduce data
    count = len(data)
    perc_sampled = 0.001
    size = int(count * perc_sampled)
    idx = np.random.choice(count, size)
    data = data[idx]

    x,y,t,p,u,v = unstack(data, axis=-1)

    # Sample/filter/add data
    is_interior = (t <= 1)
    is_exterior_1 = (t <= 2) & (t > 1)
    is_exterior_2 = (t > 2)
    is_boundary = (x == 0) | (x == 1) | (y == 0) | (y == 1)

    is_labeled = is_interior
    inputs = np.stack([x,y,t], axis=-1)
    outputs = np.stack([p], axis=-1)
    np.savez(
        wave_data_dir + '/processed_data.npz', 
        inputs=inputs, outputs=outputs, is_labeled=is_labeled,
        is_interior=is_interior, is_exterior_1=is_exterior_1, is_exterior_2=is_exterior_2,
    )

    toc = time.time()
    print("Time elapsed: ", toc - tic)