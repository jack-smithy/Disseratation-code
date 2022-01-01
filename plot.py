import numpy as np
import matplotlib.pyplot as plt
from ising_model import Ising

SIZE = 64
SIZES = [32, 64, 128]
LIVE = True

START_T = 1
END_T = 4
N_T = 11
MC_STEPS = 5e4


def multiple_sizes(sizes):
    C_vals = []
    X_vals = []

    for size in sizes:
        s = Ising(N=size, live_show=LIVE)
        results = s.simulate(np.linspace(START_T, END_T, N_T), MC_STEPS)
        T, E, M, C, X = s.make_tuple(results)
        C_vals.append(C)
        X_vals.append(X)

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Multiple lattice sizes')
    
    axs[0].plot(T, C_vals[0], label=f'N={sizes[0]}')
    axs[0].plot(T, C_vals[1], label=f'N={sizes[1]}')
    axs[0].plot(T, C_vals[2], label=f'N={sizes[2]}')
    axs[0].grid()
    axs[0].set_ylabel('C')

    axs[1].plot(T, X_vals[0])
    axs[1].plot(T, X_vals[1])
    axs[1].plot(T, X_vals[2])
    axs[1].grid()
    axs[1].set_ylabel('X')
    axs[1].set_xlabel('T/J')

    plt.show()

def plot_quantities(T, E, M, C, X):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6,4), 
                            constrained_layout=True)
    fig.suptitle(f'Ising Model size={SIZE}, MC steps={MC_STEPS}')

    axs[0,0].plot(T, M)
    axs[0,0].grid()
    axs[0,0].set_ylabel('M')

    axs[0,1].plot(T, E)
    axs[0,1].grid()
    axs[0,1].set_ylabel('E')

    axs[1,0].plot(T, C)
    axs[1,0].grid()
    axs[1,0].set_ylabel('C')

    axs[1,1].plot(T, X)
    axs[1,1].grid()
    axs[1,1].set_ylabel('X')

    plt.show()

multiple_sizes(SIZES)