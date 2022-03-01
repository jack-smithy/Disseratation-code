import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science'])

k=0.6
Is = [512, 1024, 2048]

for I in Is:
    q = np.load(f'hopfield_data/q_I={I}.npy')
    q2 = np.load(f'hopfield_data/q2_I={I}.npy')
    q4 = np.load(f'hopfield_data/q4_I={I}.npy')

    g = q4/q2**2
    Nmin, Nmax, step = int(0.05*I), int(0.22*I), int(0.005*I)
    Ns = np.arange(Nmin, Nmax, step)
    NIs = Ns/I
    fss = I**(k)*(NIs-0.14)
    plt.plot(NIs, q, label=f'I={I}')
    
plt.xlim((0.05, 0.2))
plt.ylim((0.2, 1.1))
plt.vlines(x = 0.14, ymin=0.2, ymax=1.1, linestyles='--', colors='k', lw=1)
plt.legend()
plt.ylabel(r'$\langle q^4 \rangle / \langle q^2 \rangle ^2$')
plt.xlabel('N/I')
plt.savefig('plots/hopfield_transition.pdf')
plt.show()