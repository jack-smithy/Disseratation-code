import numpy as np
import matplotlib.pyplot as plt
from ising_model import Ising

SIZE = 64
LIVE = True

START_T = 2
END_T = 3
N_T = 11
MC_STEPS = 1

s = Ising(N=SIZE, live_show=LIVE)
results = s.simulate(np.linspace(START_T, END_T, N_T), MC_STEPS)

def plot_quantities(vals):
    T, E, M, C, X = s.make_tuple(vals)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6,4), constrained_layout=True)
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

plot_quantities(results)

