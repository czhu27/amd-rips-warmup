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


def make_wave_plot(t):
	return

def make_movie(model, figs_folder, time_steps = 50, dx = .01, dt = .01):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    nx = ny = int(1 / dx)
    dt = .02

    def update(frame, fig, dt, nx, ny):
        X = np.arange(0,1,1/nx)
        Y = np.arange(0,1,1/ny)
        X, Y = np.meshgrid(X,Y)
        grid_pts = np.reshape(np.concatenate((X,Y)), (nx*ny,2))
        time_vec = np.ones((len(grid_pts),1))*dt*frame

        inputs = np.concatenate((grid_pts,time_vec), axis=1)
        soln = model.predict(inputs)
        soln = np.reshape(soln, (nx,ny))

        if len(fig.axes[0].collections) != 0:
            fig.axes[0].collections = []
            surf = fig.axes[0].plot_surface(X, Y, soln, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        else:
            surf = fig.axes[0].plot_surface(X, Y, soln, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(-.04, .04)

        fig.canvas.draw()
        return surf,

    ani = FuncAnimation(fig, update, fargs=[fig, dt, nx, ny], frames=time_steps, blit=True)
    ani.save(figs_folder + '/wave_pred.gif', writer = 'PillowWriter', fps=5)


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

def make_heatmap_animation(mat_list, save_dir):
    fig = plt.figure()
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    def animate(i):
        sns.heatmap(mat_list[i],  vmin = -3, vmax= 3, square=True, cbar=False, center=0.00, cmap=cmap)
        # plt.clf()
        # plt.close()

    anim = animation.FuncAnimation(fig, animate, frames=len(mat_list), repeat = False)

    savefile = save_dir + "/test.gif"
    pillowwriter = animation.PillowWriter(fps=5)
    anim.save(savefile, writer=pillowwriter)