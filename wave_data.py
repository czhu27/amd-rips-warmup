import numpy as np

nx, ny = 10, 10
order = 1
dt = 0.02
time_steps = int(1/dt)

def data_wave(time_steps, nx, ny, order, params):
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
	for i in range(2*time_steps):
		fname = "/home/bjt324/projectFiles/wave/forward/data/dump{:03d}.npz".format(i)
		f = open(fname, "r")
		loaded = np.load(fname, allow_pickle=True)
		lst = loaded.files
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
		rand = np.random.randint(0,len(X_w_l))
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
		X_w_ul[i,:] = np.concatenate((np.random.rand(2),np.random.randint(time_steps,2*time_steps)))	

	return X_w_l, X_w_ul, Y_l, x_flat, y_flat, t_flat, p_flat

X_w_l, X_w_ul, Y_l, x_flat, y_flat, t_flat, p_flat  = data_wave(time_steps, nx, ny, order, [1000, 1000, 1000, 1.0, 1.0, 1.0])

print("hi")