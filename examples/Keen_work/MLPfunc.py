# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:32:18 2022

@author: reghu
"""

from casadi import *


def createMultiLayerPerceptron(nInputs, nOutputs, layer,w, actfun="tanh"):
    # combine all nodes in a vector
    nodes = np.hstack((nInputs, layer, nOutputs))

    # initialize a list to hold all symbolic layers
    layer_list = []
    # initialize input layer
    x = SX.sym("x_0", nInputs, 1)

    # choose actfun
    if actfun == "tanh":
        fun = lambda x: tanh(x)
    elif actfun == "ReLU":
        fun = lambda x: fmax(x, 0)
        fun = lambda x: (sqrt(x**2 + 0.01)+x)/2
    elif actfun == "leakyReLU":
        fun = lambda x: fmax(x, 0.01*x)


    # create hidden and output layers
    for iLayer in range(1,len(nodes)):
        # create weights and biases of the ith_layer
        w = SX.sym("w_"+str(iLayer), nodes[iLayer],nodes[iLayer-1])
        b = SX.sym("b_"+str(iLayer), nodes[iLayer], 1)

        # calculate layer output and append weights and biases of the ith_layer
        if iLayer == 1:
            x_i = w @ x + b
            p_list = vertcat(w.reshape((w.numel(), 1)), b)
        else:
            x_i = w @ x_i + b
            p_list = vertcat(p_list, w.reshape((w.numel(), 1)), b)
        # add activation function for the inner layers
        if iLayer < (len(nodes)-1):
            x_i = fun(x_i)
        # add ith_layer to the list
        layer_list.append(x_i)

    # define forward pass as a symbolic function y = f(x, p)
    net_fun = Function('net', [x, p_list], [x_i])

    # define the output struct
    net_dict = {"fun":net_fun,
                "inputs":x,
                "layers":layer_list,
                "weights":p_list,
                }
    return net_dict

def assembleNarxInput(x_old, u_new, narx):
    nD_U = narx["nInputDelay"] + 1
    nD_X = narx["nOutputDelay"] + 1
    n_U = narx["nInputs"]
    n_X = narx["nOutputs"]

    x_new = vertcat(x_old[0:(nD_X * n_X)],
                    u_new,
                    x_old[(nD_X*n_X):-n_U]
                    )
    return x_new

def assembleNarxOutput(x_new, y_new, narx):
    nD_U = narx["nInputDelay"] + 1
    nD_X = narx["nOutputDelay"] + 1
    n_U = narx["nInputs"]
    n_X = narx["nOutputs"]

    y_hat = vertcat(y_new,
                    x_new[0:n_X*(nD_X-1)],
                    x_new[nD_X*n_X:]
                    )
    return y_hat