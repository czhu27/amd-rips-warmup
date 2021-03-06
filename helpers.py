import numpy as np
import tensorflow as tf
import os

def stack_unstack(X_f):
    # Needed for gradient calculations (per-column)
    xyz = tf.unstack(X_f, axis=1)
    new_X_f = tf.stack(xyz, axis=1)
    return new_X_f, xyz	

class Configs:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class GradReg:
    def __init__(self, gr_dict):
        self.region = gr_dict['region']
        self.name = gr_dict['name']
        self.vector_func = gr_dict['vector_func']
        self.is_scheduled = ('schedule' in gr_dict) and (gr_dict['schedule'] is not None)
        if self.is_scheduled:
            self.min_weight, self.max_weight = gr_dict['weight']
            self.start_epoch, self.end_epoch = gr_dict['schedule']
            self.init_weight = self.min_weight
        else:
            self.init_weight = gr_dict['weight']

def shuffle_in_parallel(mat_list):
    indices = tf.random.shuffle(np.arange(mat_list[0].shape[0]))
    for i, mat in enumerate(mat_list):
        # Handle lists of dictionaries
        if isinstance(mat, dict):
            new_mat = {}
            for k,v in mat.items():
                new_mat[k] = tf.gather(v, indices)
        else:
            new_mat = tf.gather(mat, indices)
        mat_list[i] = new_mat
    
    return mat_list

def get_delta(x):
    '''
    Gets step size of a list of grid values
    '''
    all_vals = np.sort(np.unique(x))
    dx = all_vals[1] - all_vals[0]
    x_min = all_vals[0]
    x_max = all_vals[-1]
    return dx, x_min, x_max

def unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)

def np_unstack(a, axis = 0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]

def random_rows(data, perc):
    num = len(data) * perc
    idx = np.random.choice(len(data), num)
    return data[idx]


def point_to_index(x, dx, x_min):
    n = (x - x_min) / dx
    tol = 1e-3
    n_int = round(n)
    assert abs(n - n_int) < tol, f"x={n} isn't an integer in this grid scheme"
    n = n_int
    return n

def make_empty_list_matrix(N, M):
    d = np.empty((N,M),object)
    for i in range(N):
        for j in range(M):
            d[i,j] = []
    return d

def get_p_mat(p, x, y, dx, dy, x_min, y_min, N, M):
    p_mat = make_empty_list_matrix(N,M)

    for x0,y0,p0 in zip(x,y,p):
        # x,y -> n,m
        n = point_to_index(x0, dx, x_min)
        m = point_to_index(y0, dy, y_min)
        n = min(n, N-1)
        m = min(m, M-1)
        # 
        p_mat[n,m].append(p0)

    avg = np.vectorize(lambda l: sum(l) / len(l))
    p_mat = avg(p_mat)

    return p_mat

def get_p_mat_simple(p, x, y):
	'''
	Converts vectors p, x, y into matrix p_mat.
	'''
	dx, x_min, x_max = get_delta(x)
	dy, y_min, y_max = get_delta(y)
	N = point_to_index(x_max, dx, x_min)
	M = point_to_index(y_max, dy, y_min)
	p_mat = get_p_mat(p, x, y, dx, dy, x_min, y_min, N, M)
	return p_mat

def get_p_mat_list(p_all, x_all, y_all, every_n_frames=1):
    p_mat_list = []
    T = len(p_all)
    for t, (p, x, y) in enumerate(zip(p_all, x_all, y_all)):
        if t % every_n_frames != 0:
            continue
        if t % 10 == 0:
            print(f"Generated p_mat {t}. Now {(t / T) * 100 : .0f}% complete")
        p_mat = get_p_mat_simple(p, x, y)
        p_mat_list.append(p_mat)

    p_mat_list = np.array(p_mat_list)
    return p_mat_list


# def get_p_mat_list(p_all, x_all, y_all):

# 	assert len(x_all.shape) == 2, "First axis should be time"
    
# 	T, slice_size = x_all.shape
# 	dx, x_min, x_max = get_delta(x_all[0])
# 	dy, y_min, y_max = get_delta(y_all[0])
# 	N = point_to_index(x_max, dx, x_min)
# 	M = point_to_index(y_max, dy, y_min)
# 	p_mat_list = np.zeros((T, N, M))
# 	for t, p, x, y in enumerate(zip(p_all, x_all, y_all)):
# 		p_mat = get_p_mat(p_all[t], x_all[t], y_all[t], dx, dy, x_min, y_min, N, M)
# 		p_mat_list[t,:,:] = p_mat
    
# 	return p_mat_list

def find(name, path):
    res = []
    for root, dirs, files in os.walk(path):
        if name in files:
            res.append(os.path.join(root, name))
    return res