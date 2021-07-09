import matplotlib.pyplot as plt
import numpy as np

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