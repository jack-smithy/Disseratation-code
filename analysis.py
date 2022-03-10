import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science'])
plt.rcParams['figure.figsize'] = 3,3
plt.rcParams['figure.constrained_layout.use'] = True

k=0.6
Is = [512, 1024, 2048]

for I in Is:
    q = np.load(f'hopfield_data/q_I={I}.npy')
    q2 = np.load(f'hopfield_data/q2_I={I}.npy')
    q4 = np.load(f'hopfield_data/q4_I={I}.npy')

    g = q4/q2**2
    X = q2 - q**2

    Nmin, Nmax, step = int(0.05*I), int(0.22*I), int(0.005*I)
    Ns = np.arange(Nmin, Nmax, step)
    alphas = Ns/I
    fss = I**k*(alphas-0.12)
    plt.plot(fss, q, label=f'$I={I}$')
    
plt.xlim((-2.5, 2.5))
plt.ylim((0.3, 1.05))
#plt.vlines(x = 0.12, ymin=0, ymax=0.8, linestyles='--', colors='k', lw=1)
plt.legend()
plt.ylabel(r'$q$')
plt.xlabel(r'$I^{1/\nu} (\alpha - \alpha_C)$')
plt.savefig(f'plots/hopfield_cumulant_collapse.pdf')
plt.show()
