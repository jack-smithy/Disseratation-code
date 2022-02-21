import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')

x = np.linspace(-8, 8, 201)

y1 = np.exp(-1/2*(x+8))

y2 = (1/(np.exp(8)))*np.exp(-(x-8))

plt.plot(x, y1, color='k')
plt.plot(x[100:], y2[100:], '--', color='k')
plt.hlines(y=1, xmin=-8, xmax=0, colors='k', linestyles='--')
plt.xlim((-8,8))
plt.xticks(ticks=[-8, -4, 0, 4, 8])
plt.ylim((0, 1.1))
plt.xlabel(r'$\Delta E$')
plt.ylabel('Acceptance ratio')
plt.savefig('plots/acceptance_ratio.pdf')
plt.show()