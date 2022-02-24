import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use('science')
#plt.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.grid"] = False

#Simulation params
Ns = [16, 24, 32]
numNs = len(Ns)

numTs = 61
numT2s = 10
Ts = np.linspace(1, 4, numTs)
T2s = np.linspace(2.2, 2.4, numT2s)

#Init data storage
M = np.zeros(shape=(numNs, numTs))
M2 = np.zeros(shape=(numNs, numTs))
M4 = np.zeros(shape=(numNs, numTs))

MZ2 = np.zeros(shape=(numNs, numT2s))
MZ4 = np.zeros(shape=(numNs, numT2s))

E = np.zeros(shape=(numNs, numTs))
E2 = np.zeros(shape=(numNs, numTs))

#Read and store data
for i, N in enumerate(Ns):
    for j, T in enumerate(Ts):
        mm = np.load(f'data/Mag{int(j)}_N={N}.npy')
        ee = np.load(f'data/Ene_T{int(j)}_N={N}.npy')

        M[i, j] = np.mean(mm)
        E[i, j] = np.mean(-abs(ee))
        M2[i,j] = np.mean(np.power(mm, 2))
        E2[i,j] = np.mean(np.power(ee, 2))
        M4[i,j] = np.mean(np.power(mm, 4))

'''
###Cumulant plots###
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), sharey=False)
for i, n in enumerate(Ns):
    g = 0.5*(3-M4[i,:]/M2[i,:]**2)
    axs[0].plot(Ts, g, label=f'$L={n}$')
axs[0].set_xlim(left=1.5, right=3)
axs[0].set_ylim(bottom=0, top=1.1)
axs[0].axvline(x=2.269, ymin=0, ymax=1.1, color='black', ls='--')
axs[0].set_ylabel('$g$')
axs[0].set_xlabel('$T$')

for i, n in enumerate(Ns):
    g = 0.5*(3-M4[i,:]/M2[i,:]**2)
    axs[1].plot(n*(Ts-2.269), g, label=f'$L={n}$')
axs[1].set_xlim(left=-10, right=10)
axs[1].set_ylim(bottom=0.7, top=1.05)
axs[1].set_ylabel('$g$')
axs[1].set_xlabel(r'$L^{1/\nu}(T-T_c)$')
axs[1].legend()
#plt.savefig('plots/data_collapse.pdf')
#plt.show()
'''

###E-M plots###
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
for i, n in enumerate(Ns):
    ene = E[i,:]
    axs[0].plot(Ts, ene)
axs[0].set_xlim(left=1, right=4)
axs[0].set_ylim(bottom=-2.1, top=-0.5)
#axs[0].axvline(x=2.269, ymin=-2.1, ymax=-0.5, color='black', ls='--')
axs[0].set_ylabel('$E$')
axs[0].set_xlabel('$T$')

for i, n in enumerate(Ns):
    mag = M[i,:]
    axs[1].plot(Ts, mag, label=f'$L={n}$')
axs[1].set_xlim(left=1, right=4)
axs[1].set_ylim(bottom=0, top=1.1)
#axs[1].axvline(x=2.269, ymin=0, ymax=1.1, color='black', ls='--')
axs[1].set_ylabel('$m$')
axs[1].set_xlabel('$T$')
axs[1].legend()
#plt.savefig('plots/ene_mag.pdf')
#plt.show()

###c-X plots###
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
for i, n in enumerate(Ns):
    ee = E2[i,:] - (E[i,:])**2
    c = ee/(Ts**2*N)
    axs[0].plot(Ts, c*10**4)
axs[0].set_xlim(left=1, right=4)
axs[0].set_ylim(bottom=0, top=2)
#axs[0].axvline(x=2.269, ymin=0, ymax=0.0002, color='black', ls='--')
axs[0].set_ylabel(r'$c \times 10 ^ 4$')
axs[0].set_xlabel('$T$')

for i, n in enumerate(Ns):
    mm = M2[i,:] - (M2[i,:])**2
    X = mm*N/Ts
    axs[1].plot(Ts, X, label=f'$L={n}$')
axs[1].set_xlim(left=1, right=4)
axs[1].set_ylim(bottom=0, top=3.8)
formatter = mticker.ScalarFormatter(useMathText=True)
axs[1].yaxis.set_major_formatter(formatter)
#axs[1].axvline(x=2.269, ymin=0, ymax=3.8, color='black', ls='--')
axs[1].set_ylabel(r'$\chi$')
axs[1].set_xlabel('$T$')
axs[1].legend()
plt.savefig('plots/c-X.pdf')
plt.show()
