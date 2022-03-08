import numpy as np
import matplotlib.pyplot as plt

q11 = np.load('hopfield_data/q_I=1024.npy')
q21 = np.load('hopfield_data/q2_I=1024.npy')
q41 = np.load('hopfield_data/q4_I=1024.npy')

q12 = np.load('hopfield_data/q_I=2048.npy')
q22 = np.load('hopfield_data/q2_I=2048.npy')
q42 = np.load('hopfield_data/q4_I=2048.npy')

q13 = np.load('hopfield_data/q_I=3072.npy')
q23 = np.load('hopfield_data/q2_I=3072.npy')
q43 = np.load('hopfield_data/q4_I=3072.npy')

I = 2048
Nmin, Nmax, step = int(0.05*I), int(0.22*I), int(0.005*I)
Ns = np.arange(Nmin, Nmax, step)

NIs = Ns/I

g1 = q41/q21**2
g2 = q42/q22**2
g3 = q43/q23**2


plt.plot(NIs, g1)
plt.plot(NIs, g2)
plt.plot(NIs, g3)
#plt.ylim(0, 2.9)
plt.xlim(0.05, 0.2)
#plt.vlines(x=0.138, ymin=0, ymax=2.9, linestyles='--', colors='k')
plt.show()
