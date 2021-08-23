'''
    Machine learning python project 2
    Michal Keren
    Itamar Eyal
Libraries: numpy, matplotlib.
'''

'''
    IMPORTS
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_random_state
import sys


'''
    DEFINES
'''
x_train = np.array([[0,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1]])

ETA = 2
ITERATIONS = 2000
TURNS = 100

'''
    IMPLEMENTATIONS
'''

def logistic_sigmoid(x):
    exp = np.exp(-x)
    return np.divide(1,(np.add(1,exp)))

def error(y,t):
    return np.sum(np.power( np.add(y, - t) , 2 )) / 8

def build_t_train(x):
    sums = np.sum(x, axis=1)
    t = np.ones_like(sums)
    t[np.mod(sums, 2)==0] = 0
    return t


def build_w_HL(cell_num):
        return np.random.normal(0, 1, size=(cell_num, 4))  # each row is for a cell.


def build_w_out(cell_num):
    return np.random.normal(0, 1, size=(cell_num+1))  # each row is for a cell.

def get_delta_output(t, y):
    h_prime = np.multiply(y, np.add(1, -y))
    delta_out = np.multiply(h_prime, np.add(y, -t))
    return delta_out


def get_delta_hl(delta, z, w_row,cell_num):
    delta_resized = np.tile(delta, (1, cell_num))
    w_resized = np.tile(w_row.T, (8, 1))

    h_prime = np.multiply(z, np.add(1, -z))
    h_w = np.multiply(h_prime, w_resized)
    delta_hl = np.multiply(delta_resized, h_w)
    return delta_hl

def get_ai(z, row_w):
    return np.dot(row_w[1:], z.T) + row_w[0] * np.ones(shape=(1, 8))

def parity_3(x, t,cell_num):
    w_HL = build_w_HL(cell_num)
    w_out= build_w_out(cell_num)
    a = np.zeros(shape=(8, cell_num))  # holds x*w
    E = np.zeros(ITERATIONS)  # holds MSE by iteration for the turn

    for iter in range(ITERATIONS):
        # fill a mat for every node of the hidden layer
        for i in range(cell_num):
            a[:, i] = get_ai(x, w_HL[i])
        z = logistic_sigmoid(a)

        # fill a mat for output node
        a_out = get_ai(z, w_out)
        y = logistic_sigmoid(a_out).T

        # calculate MSE
        E[iter] = error(y, t)

        delta_out = get_delta_output(t, y)
        delta_hl = get_delta_hl(delta_out, z, w_out[1:],cell_num)

        # update weights for output node
        w_out[1:] = w_out[1:] - ETA * np.sum(np.multiply(delta_out, z), axis=0)
        # update weights for hidden layer nodes
        for node in range(cell_num):
            delta_node=delta_hl[:,node].reshape((8,1))
            w_HL[node][1:] = np.add(w_HL[node][1:], - ETA * np.sum(np.multiply(delta_node, x), axis=0))
    return E

'''
    EXECUTION - PART A & B
'''
t_train = build_t_train(x_train)
t_train= np.reshape(t_train, (8,1))
E_mat = np.zeros(shape=(TURNS,ITERATIONS))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Parity-3 using 1 HL')
for part in ['A','B']:
    if part =='A':
        cell_num = 3
        ax=ax1
        title = "Part A: HL containing 3 cells"
    else:
        cell_num = 6
        ax=ax2
        title= "Part B: HL containing 6 cells"
    for turn in range(TURNS):
        # calls parity3 to get MSE vector
        E_mat[turn] = parity_3(x_train,t_train,cell_num)
    # mean of MSE of all turns by iteration
    results = np.mean(E_mat, axis=0)

    # plot
    ax.plot(results)
    ax.set(xlabel="iteration", ylabel="MSE")
    ax.set_title(title)
plt.show()

