import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation

import io
import seaborn as sns
from matplotlib import animation

def plot_data_2D(X_l, X_ul, save_dir):
    '''
    Plot input data (2D)
    '''
    plt.scatter(X_l[:,0], X_l[:,1], s=2, color='b')
    plt.scatter(X_ul[:,0], X_ul[:,1], s=2, color='g')
    plt.legend()
    plt.savefig(save_dir + "/data")
    plt.clf()

def plot_gridded_functions(model, f, lb, ub, tag, folder="figs"):
    n1d = 101
    npts = n1d*n1d
    x0 = np.linspace(lb, ub, n1d)
    x1 = np.linspace(lb, ub, n1d)
    x0_g, x1_g = np.meshgrid(x0, x1)

    # Compute true function values
    f_true = f(x0_g, x1_g)

    # Compute ML function values
    ml_input = np.zeros((npts, 2))
    ml_input[:,0] = x0_g.flatten()
    ml_input[:,1] = x1_g.flatten()
    ml_output = model.predict(ml_input)
    f_ml = np.reshape(ml_output, (n1d, n1d), order = 'C')

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.tight_layout()
    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(x0_g, x1_g, f_true, cmap=cm.coolwarm)
    ax.set_title('True')
    #plt.savefig('figs/true' + str(tag) + '.png')

    ax = fig.add_subplot(222, projection='3d')
    ax.plot_surface(x0_g, x1_g, f_ml, cmap=cm.coolwarm)
    ax.set_title('ML')
    #plt.savefig('figs/ml' + str(tag) + '.png')

    ax = fig.add_subplot(223, projection='3d')
    ax.plot_surface(x0_g, x1_g, np.abs(f_ml - f_true), cmap=cm.coolwarm)
    ax.set_title('|True - ML|')
    #plt.savefig('figs/diff' + str(tag) + '.png')
    plt.savefig(folder + '/all' + str(tag) + '.png')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def make_wave_plot(model, t, f_true, figs_folder, tag):
    n1d = 101
    lb = 0
    ub = 1
    npts = n1d*n1d
    x0 = np.linspace(lb, ub, n1d)
    x1 = np.linspace(lb, ub, n1d)
    x0_g, x1_g = np.meshgrid(x0, x1)

    # Compute true function values
    #f_true = f(x0_g, x1_g)

    # Compute ML function values
    ml_input = np.zeros((npts, 3))
    ml_input[:,0] = x0_g.flatten()
    ml_input[:,1] = x1_g.flatten()
    ml_input[:,2] = t*np.ones((npts))
    ml_output = model.predict(ml_input)
    f_ml = np.reshape(ml_output, (n1d, n1d), order = 'C')

    fig = plt.figure()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.tight_layout()
    # ax = fig.add_subplot(221, projection='3d')
    # ax.plot_surface(x0_g, x1_g, f_true, cmap=cm.coolwarm)
    # ax.set_title('True')
    #plt.savefig('figs/true' + str(tag) + '.png')

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x0_g, x1_g, f_ml, cmap=cm.coolwarm)
    ax.set_title('ML')
    #plt.savefig('figs/ml' + str(tag) + '.png')

    # ax = fig.add_subplot(223, projection='3d')
    # ax.plot_surface(x0_g, x1_g, np.abs(f_ml - f_true), cmap=cm.coolwarm)
    # ax.set_title('|True - ML|')
    #plt.savefig('figs/diff' + str(tag) + '.png')
    plt.savefig(figs_folder + '/all' + str(tag) + '.png')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def make_movie(model, figs_folder, time_steps = 100, dx = .01, dt = .01):
    #Create figure for movie and init constants
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    nx = ny = int(1 / dx)
    dt = .01

    def update(frame, fig, dt, nx, ny):
        #Creates inputs
        X = np.arange(0,1,1/nx)
        Y = np.arange(0,1,1/ny)
        X_g, Y_g = np.meshgrid(X,Y)
        grid_pts = np.reshape(np.concatenate((X_g,Y_g)), (nx*ny,2))
        time_vec = np.ones((len(grid_pts),1))*dt*frame
        #Runs inputs through model for soln
        X_t = np.reshape(X_g, (nx*ny,1))
        Y_t = np.reshape(Y_g, (nx*ny,1))
        inputs = np.concatenate((X_t, Y_t,time_vec), axis=1)
        soln = model.predict(inputs)
        print(soln)
        soln = np.reshape(soln, (nx,ny))
        #Clears current fig and draws surface
        if len(fig.axes[0].collections) != 0:
            fig.axes[0].collections = []
            surf = fig.axes[0].plot_surface(X_g, Y_g, soln, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        else:
            surf = fig.axes[0].plot_surface(X_g, Y_g, soln, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-.75,.75)

        fig.canvas.draw()
        return surf,
    #Loops through update to create animation
    ani = FuncAnimation(fig, update, fargs=[fig, dt, nx, ny], frames=time_steps, blit=True)
    ani.save(figs_folder + '/wave_pred2.gif', writer = 'PillowWriter', fps=10)


def plot_data_dist(x,y):
    '''
    Plots flattened (1D) x, y arrays
    '''
    x_c, y_c = x[4::9], y[4::9]
    plt.scatter(x,y, s=0.1, alpha=0.3)
    plt.scatter(x_c,y_c, s=0.1, c='r')
    plt.show()
    #plt.savefig('test.png', dpi=1000)

# def plot_heatmap(x,y,p):
#     plt.figure(figsize=(8, 8))
#     cmap = cm.get_cmap('Reds')
#     plt.scatter(x,y, s=3.7, marker='s', c=p, cmap=cmap)
#     plt.savefig('test.png', dpi=500)
#     plt.show()

def make_heatmap_animation(mat_list, save_dir, R=None, fps = 10):
    '''
    '''
    assert len(mat_list.shape) == 3, "Needs to be an array, not a list. An array of shape (T,N,M)"
    avg = np.mean(mat_list)
    std = np.std(mat_list)
    color_spread = 1.5
    if R is None:
        # Deduce R from matrix
        R = np.max([np.abs(avg + color_spread * std), np.abs(avg - color_spread * std)])

    fig = plt.figure()
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    def animate(i):
        if i % 3 == 0:
            print(f"Animation frame: {i}. Now {i / len(mat_list) * 100 :.0f}% done with animation.")
        sns.heatmap(mat_list[i],  vmin = -R, vmax= R, square=True, cbar=False, center=0.00, cmap=cmap)
        # plt.clf()
        # plt.close()

    anim = animation.FuncAnimation(fig, animate, frames=len(mat_list), repeat = False)

    savefile = save_dir + "/heatmap.gif"
    pillowwriter = animation.PillowWriter(fps=fps)
    anim.save(savefile, writer=pillowwriter)

def wave_model_heatmap(model, figs_folder, x_min=0, x_max=1, dx = 0.05, t_min=0, t_max=1, dt=None):
    print("Making model heatmap plot...")
    nx = int((x_max - x_min) / dx)
    if dt is None:
        dt = dx
    nt = int((t_max - t_min) / dt)
    lb=0; ub=1
    x_lin = np.linspace(lb, ub, nx)
    y_lin = np.linspace(lb, ub, nx)
    t_lin = np.linspace(0, 2, nt)
    t_g, x_g, y_g = np.meshgrid(t_lin, x_lin, y_lin, indexing="ij")
    def flatten_end(arr):
        return arr.reshape(arr.shape[0], -1)

    x = flatten_end(x_g)
    y = flatten_end(y_g)
    t = flatten_end(t_g)

    X = np.stack([x,y,t], axis=-1)
    p_mat_arr = np.zeros((len(t_lin), len(x_lin), len(y_lin)))
    for i, t_min in enumerate(t):
        Y_pred = model.predict(X[i])
        Y_pred = Y_pred.reshape(len(x_lin), len(y_lin))
        p_mat_arr[i] = Y_pred
    
    from plots import make_heatmap_animation
    make_heatmap_animation(p_mat_arr, figs_folder, fps = 2)

    print("Finished model heatmap plot")

