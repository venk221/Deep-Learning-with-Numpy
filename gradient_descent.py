import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X = [0.5, 2.5]
Y = [0.2, 0.9]

def function(w,b,x):
    return 1.0 / (1.0 + np.exp(-(w*x) + b))

def error(w,b):
    err = 0.0
    for (x,y) in zip(X,Y):
        fx = function(w,b,x)
        err = err + 0.5 * (fx - y)**2

    return err

def grad_w(w,b,x,y):
    fx = function(w,b,x)
    return (fx-y)*fx*(1-fx)*x

def grad_b(w,b,x,y):
    fx = function(w,b,x)
    return (fx-y)*fx*(1-fx)


def do_grad_descent():
    w_list = []
    b_list = []
    error_list = []
    w, b, lr, max_epochs = -0, -0, 0.1, 100
    for i in range(max_epochs):
        dw, db = 0.0, 0.0
        for (x,y) in zip(X,Y):
            dw = dw + grad_w(w,b,x,y)
            db = db + grad_b(w,b,x,y)
        w = w - lr*dw
        b = b - lr*dw
        Error = error(w,b)
        w_list.append(w)
        b_list.append(b)
        error_list.append(Error)

    error_list = np.array(error_list)
    w_list = np.array(w_list)
    b_list = np.array(b_list)

    surf = ax.plot_surface(w_list, b_list, np.expand_dims(error_list,axis=1), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    
do_grad_descent()

