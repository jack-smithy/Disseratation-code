import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
plt.style.use('science')

Xmin, Xmax, samples = -2, 2, 100

Xs = np.linspace(Xmin, Xmax, samples)
tanhXs = np.tanh(0.8*Xs)
tanh4Xs = np.tanh(2*Xs)

fig, axs = plt.subplots(nrows = 1, ncols=2 , sharey=True, figsize = (6,3))
axs[0].plot(Xs, tanhXs, label=r'$\tanh (Jqm \beta)$')
axs[0].plot(Xs, Xs, label='m')
axs[0].set_xlim((Xmin, Xmax))
axs[0].set_yticklabels([])
axs[0].set_xticklabels([])

axs[1].plot(Xs, tanh4Xs, label=r'$\tanh (Jqm \beta)$')
axs[1].plot(Xs, Xs, label='m')
axs[1].set_xlim((Xmin, Xmax))
axs[1].set_yticklabels([])
axs[1].set_xticklabels([])
axs[1].legend()

plt.ylim((-1.5, 1.5))
plt.savefig('plots/mft.pdf')
plt.show()