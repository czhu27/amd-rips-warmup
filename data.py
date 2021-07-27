from seaborn.matrix import heatmap
from plots import make_heatmap_animation
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
import io
import shutil

from helpers import get_p_mat_list, unstack

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

	ml_input = np.zeros((npts, 2))
	ml_input[:,0] = x0_g.flatten()
	ml_input[:,1] = x1_g.flatten()
	# ml_input[:,2] = t*np.ones((npts))
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

def compute_error_wave(model, test_set):
	#test_set can be int_test or ext_test
	#formatting
	f_true = np.reshape(test_set[:,3],(len(test_set[:,3]),1))
	x = np.reshape(test_set[:,0],(len(test_set[:,0]),1))
	y = np.reshape(test_set[:,1],(len(test_set[:,1]),1))
	t = np.reshape(test_set[:,2],(len(test_set[:,2]),1))

	#Computes model and finds difference with sim
	ml_input = np.concatenate((x,y,t),axis=1)
	ml_output = model.predict(ml_input)
	f_ml = np.reshape(ml_output, (len(f_true), 1))
	error = np.sqrt(np.mean(np.square(f_ml - f_true)))
	return error

def error_time(model, int_test, ext_test, figs_folder, tag):
	#Takes all data, int and ext
	fig, ax = plt.subplots()
	starter_iter = int_test[0,2]
	int_times, total_int = np.unique(int_test[:,2], return_counts=True)
	ext_times, total_ext = np.unique(ext_test[:,2], return_counts=True)
	tf = ext_test[-1,2]
	total_int = total_int[0]
	total_ext = total_ext[0]
	sample_step = int_times[1] - int_times[0]
	all_test = np.concatenate((int_test, ext_test))
	time_axis = np.concatenate((int_times, ext_times))
	time_steps = len(time_axis)
	error_axis = []
	for i in range(0,time_steps):
		if i > tf/sample_step:
			#formatting
			f_true = np.reshape(all_test[i*total_int:(i+1)*total_int,3],(total_int,1))
			x = np.reshape(all_test[i*total_int:(i+1)*total_int,0],(total_int,1))
			y = np.reshape(all_test[i*total_int:(i+1)*total_int,1],(total_int,1))
			t = np.reshape(all_test[i*total_int:(i+1)*total_int,2],(total_int,1))

			#Computes model and finds difference with sim
			ml_input = np.concatenate((x,y,t),axis=1)
			ml_output = model.predict(ml_input)
			f_ml = np.reshape(ml_output, (len(f_true), 1))
			error = np.sqrt(np.mean(np.square(f_ml - f_true)))
			error_axis.append(error)

		else:
			#formatting
			f_true = np.reshape(all_test[i*total_ext:(i+1)*total_ext,3],(total_ext,1))
			x = np.reshape(all_test[i*total_ext:(i+1)*total_ext,0],(total_ext,1))
			y = np.reshape(all_test[i*total_ext:(i+1)*total_ext,1],(total_ext,1))
			t = np.reshape(all_test[i*total_ext:(i+1)*total_ext,2],(total_ext,1))

			#Computes model and finds difference with sim
			ml_input = np.concatenate((x,y,t),axis=1)
			ml_output = model.predict(ml_input)
			f_ml = np.reshape(ml_output, (len(f_true), 1))
			error = np.sqrt(np.mean(np.square(f_ml - f_true)))
			error_axis.append(error)

	ax.plot(time_axis, error_axis)
	ax.set_xlim(.25,2)

	plt.savefig(figs_folder + str(tag) + '.png')
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	return -10

def load_data(dump_file):
	'''
	Loads data from dump file
	'''
	stuff = np.load(dump_file)
	pts = stuff['pts']
	bound = stuff['bound']
	return pts, bound

def load_slices(dump_file):
	'''
	Loads data from dump file
	'''
	stuff = np.load(dump_file)
	pts = stuff['pts']
	bound = stuff['bound']
	return pts, bound

def process_wave_data_sample(wave_data_dir, params):
	# Seed for reproducibility
	np.random.seed(params["seed"])

	tic = time.time()

	# TODO: This is bad
	label_int = 1
	label_ext = 0
	# End bad stuff
	tf = params["tf"]
	dt = params["dt"]
	T = int(tf / dt) + 1

	#Determine step_size
	if "sample_step" in params:
		step_size = int(params["sample_step"]/dt)
		assert params["sample_step"]/dt - step_size < 0.001, "Sample step not a multiple of dt"
	else:
		step_size = 1

	#Initialize datasets
	interior = np.zeros((0,6), dtype = np.float32)
	exterior = np.zeros((0,6), dtype = np.float32)
	int_bound = np.zeros((0,6), dtype = np.float32)
	ext_bound = np.zeros((0,6), dtype = np.float32)
	int_test = np.zeros((0,6), dtype = np.float32)
	ext_test = np.zeros((0,6), dtype = np.float32)

	#x_all = np.zeros(int(tf/params["sample_step"]), )
	x_all = None
	y_all = None
	p_all = None

	#Read in files and sort
	start_iter = 10*int(params["sample_step"]/params['dt'])
	for i in range(start_iter, int(tf/params["dt"]), int(params["sample_step"]/params['dt'])):
		pts, boundaries = load_data(wave_data_dir + "/dumps/dump{:03d}.npz".format(i))
		#If interior
		if i == start_iter:
			x_all = np.zeros((0,pts.shape[0]+boundaries.shape[0]))
			y_all = np.zeros((0,pts.shape[0]+boundaries.shape[0]))
			p_all = np.zeros((0,pts.shape[0]+boundaries.shape[0]))
			#u_all
			#v_all
		all_pts = np.concatenate((pts, boundaries))
		x_all = np.append(x_all, np.array([all_pts[:,0]]), axis=0)
		y_all = np.append(y_all, np.array([all_pts[:,1]]), axis=0)
		p_all = np.append(p_all, np.array([all_pts[:,3]]), axis=0)
		if i <= 1/params["dt"]:
			#Separate boundary, interior points, test set
			num_pts = int(params["data_percents"][0][0]*pts.shape[0])
			num_bound = int(params["data_percents"][0][1]*boundaries.shape[0])
			num_test = int(params["data_percents"][2][0]*pts.shape[0])
			indices_pts = np.random.randint(0,pts.shape[0], num_pts)
			interior = np.append(interior, pts[indices_pts,:], axis=0)
			indices_bound = np.random.randint(0,boundaries.shape[0], num_bound)
			int_bound = np.append(int_bound, boundaries[indices_bound,:], axis=0)
			indices_test = np.random.randint(0,pts.shape[0] + boundaries.shape[0], num_test)
			int_test = np.append(int_test, all_pts[indices_test,:], axis = 0)
		#If exterior
		else:
			#Separate boundary, interior points, test set
			num_pts = int(params["data_percents"][1][0]*pts.shape[0])
			num_bound = int(params["data_percents"][1][1]*boundaries.shape[0])
			num_test = int(params["data_percents"][2][1]*pts.shape[0])
			indices_pts = np.random.randint(0,pts.shape[0], num_pts)
			exterior = np.append(exterior, pts[indices_pts,:], axis=0)
			indices_bound = np.random.randint(0,boundaries.shape[0], num_bound)
			ext_bound = np.append(ext_bound, boundaries[indices_bound,:], axis=0)
			indices_test = np.random.randint(0,pts.shape[0] + boundaries.shape[0], num_test)
			ext_test = np.append(ext_test, all_pts[indices_test,:], axis = 0)

	#Delete dump files
	shutil.rmtree(wave_data_dir + "/dumps/")

	#Randomly sample labeled and unlabeled data on interior
	num_int_label = int(interior.shape[0]*label_int)
	perm_int = np.random.permutation(interior)
	int_label = perm_int[0:num_int_label,:]
	int_unlabel = perm_int[num_int_label:,:]

	#Randomly sample labeled and unlabeled data on exterior
	if tf > 1:
		num_ext_label = int(exterior.shape[0]*label_ext)
		perm_ext = np.random.permutation(exterior)
		ext_label = perm_ext[0:num_ext_label,:]
		ext_unlabel = perm_ext[num_ext_label:,:]
	else:
		ext_label = np.array([])
		ext_unlabel = np.array([])
		ext_bound = np.array([])
	
	if params["heatmap"]:
		# Make crude heatmap
		print("Making p_mat_list (for heatmap animation)...")
		heatmap_dt = 0.02
		heatmap_seconds_per_second = 0.2
		heatmap_step_size = int(heatmap_dt / dt)
		if heatmap_step_size <= step_size:
			every_n_frames = 1
		else:
			every_n_frames = int(heatmap_step_size / step_size)
		p_mat_list = get_p_mat_list(p_all, x_all, y_all, every_n_frames=every_n_frames)
		print("Making heatmap animation...")
		fps = heatmap_seconds_per_second / heatmap_dt
		make_heatmap_animation(p_mat_list, save_dir=wave_data_dir, fps=fps)
		print("Finished making heatmap animation")

	np.savez(
		wave_data_dir + '/processed_data.npz', 
		int_label = int_label, int_unlabel = int_unlabel, int_bound = int_bound, int_test = int_test, 
		ext_label = ext_label, ext_unlabel = ext_unlabel, ext_bound = ext_bound, ext_test = ext_test
	)

	toc = time.time()
	print("Time elapsed: ", toc - tic)

def process_wave_data(wave_data_dir, params):
	print("Processing wave data...")
	tic = time.time()

	# TODO: This is bad
	tf = params["tf"]
	dt = params["dt"]
	T = int(tf / dt) + 1

	# Compute step_size (how many timesteps to jump by while loading)
	if "sample_step" in params:
		step_size = int(params["sample_step"]/dt)
		assert abs(params["sample_step"]/dt - step_size) < 0.001, "Sample step not a multiple of dt"
	else:
		step_size = 1

	for i in range(0, int(tf/params["sample_step"]), int(1/params["sample_step"])):
		pts, bound, test_pts = load_data(wave_data_dir + "/dumps/dump{:03d}.npz".format(i))
	
	slice_size = len(x)

	num_slices = T // step_size

	x_all, y_all, p_all, u_all, v_all = [np.zeros((num_slices, slice_size)) for i in range(5)]
	t_all = np.zeros(num_slices)


	print("Loading data.")
	for i in range(num_slices): 
		dump_file = wave_data_dir + "/dumps/dump{:03d}.npz".format(i * step_size)
		x,y,p,u,v = load_data(dump_file)
		x_all[i, :] = x
		y_all[i, :] = y
		p_all[i, :] = p
		u_all[i, :] = u
		v_all[i, :] = v
		t_all[i] = dt * i * step_size
	print("Loaded data.")

	# Make crude heatmap
	print("Making p_mat_list (for heatmap animation)...")
	heatmap_dt = 0.02
	heatmap_seconds_per_second = 0.2
	heatmap_step_size = int(heatmap_dt / dt)
	every_n_frames = int(heatmap_step_size / step_size)
	p_mat_list = get_p_mat_list(p_all, x_all, y_all, every_n_frames=every_n_frames)
	print("Making heatmap animation...")
	fps = heatmap_seconds_per_second / heatmap_dt
	make_heatmap_animation(p_mat_list, save_dir=wave_data_dir, fps=fps)
	print("Finished making heatmap animation")

	x = x_all.flatten()
	y = y_all.flatten()
	p = p_all.flatten()
	u = u_all.flatten()
	v = v_all.flatten()
	t = np.repeat(t_all, slice_size)

	print("Sampling data...")
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

	print("Formatting data...")

	is_labeled = is_interior
	inputs = np.stack([x,y,t], axis=-1)
	outputs = np.stack([p], axis=-1)

	print("Saving data...")
	np.savez(
		wave_data_dir + '/processed_data.npz', 
		inputs=inputs, outputs=outputs, is_labeled=is_labeled,
		is_interior=is_interior, is_exterior_1=is_exterior_1, is_exterior_2=is_exterior_2,
	)

	toc = time.time()
	print("Finished processing wave data.", "Time elapsed: ", toc - tic)
