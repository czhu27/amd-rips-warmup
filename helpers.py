import numpy as np

class Configs:
	def __init__(self, **entries):
		self.__dict__.update(entries)

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
        # 
        p_mat[n,m].append(p0)

    avg = np.vectorize(lambda l: sum(l) / len(l))
    p_mat = avg(p_mat)

    return p_mat