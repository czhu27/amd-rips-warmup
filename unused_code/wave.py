import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt









# cmap = sns.color_palette("coolwarm", as_cmap=True)

# p_mat_list = []

# for t in range(0, 5*500, 30): 
#     dump_file = "data/wave/dump{:03d}.npz".format(t)
#     x,y,p,u,v = load_data(dump_file)

#     dx, x_min, x_max = get_delta(x)
#     dy, y_min, y_max = get_delta(y)

#     N = point_to_index(x_max, dx, x_min) + 1
#     M = point_to_index(y_max, dy, y_min) + 1
    
#     p_mat = get_p_mat(p, x, y, dx, dy, x_min, y_min, N, M)
#     p_mat_list.append(p_mat)

#     # sns.heatmap(p_mat, vmax=.8, square=True, cbar=True, center=0.00, cmap=cmap)
#     # plt.show()