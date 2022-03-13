import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science'])
#plt.rcParams['figure.figsize'] = 3,3
#plt.rcParams['figure.constrained_layout.use'] = True

k=-0.1
alpha_c = 0.143
Ns = [256, 512, 1024]
alphas = np.linspace(0.04, 0.32, 20)

for N in Ns:
    q = np.load(f'hopfield_data/q_N={N}.npy')
    q2 = np.load(f'hopfield_data/q2_N={N}.npy')
    q4 = np.load(f'hopfield_data/q4_N={N}.npy')

    g = 1-q4/(3*q2**2)
    X = (q2 - q**2)/alphas

    fss = N**k*(alphas-alpha_c)
    plt.plot(alphas, q,  label=f'N={N}')

plt.xlim((0.05, 0.3))
plt.ylim((0.25, 1.05))
#plt.vlines(x = 0, ymin=0.65, ymax=0.67, linestyles='--', colors='k', lw=1)
plt.legend()
plt.ylabel(r'$q$')
plt.xlabel(r'$\alpha$')
# plt.savefig(f'plots/hopfield_cumulant_collapse.pdf')
plt.show()
