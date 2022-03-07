import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

plt.style.use('science')
plt.rcParams['figure.figsize'] = 6,3
plt.rcParams['figure.constrained_layout.use'] = True

class Ising(object):
    def __init__(self, N, T=1.0, f=0.2, config='ones', live_show=False):
        """
        Initialises the lattice with the necessary attributes.

        Parameters
        ----------
        N : int
            Size of the lattice side
        T : float, optional
            Temparature of the system, by default 1J
        f : float, optional
            [description], by default 0.2
        live_show : bool
            Show a live snapshot of the lattice, by default False
        """
        self.N=N
        self.T=T
        self.f=f
        self.live_show = live_show

        self.setup_lattice(config)

        if live_show == True:
            self.setup_plotting()

    def setup_lattice(self, config):
        """
        Setup the lattice with a configuration of all up spins.
        """
        if config=='ones':
            self.lattice = np.ones([self.N, self.N])
        elif config=='random':
            self.lattice = np.random.choice([-1, 1], size=[self.N, self.N])
        else:
            print('Not a recognised configuration: must be "random" or "ones". Defaulting to "ones".')
            self.lattice = np.ones([self.N, self.N])

    def setup_plotting(self):
        """
        Setup a figure for interactive plotting.
        """
        plt.close('all')
        self.live = plt.figure(figsize=(4, 4))
        plt.ion()
        plt.axis('off')
        plt.show()

    def plot_lattice(self, thermalising, name=None, save=False):
        """
        Plot the lattice spins on an NxN grid.

        Parameters
        ----------
        thermalising : bool
            Shows the lattice in a different colour when the system is thermalising
        """
        X, Y = np.meshgrid(range(self.N), range(self.N))
        cm = plt.cm.Greys

        if thermalising:
            cm = plt.cm.Greys
        plt.gca().set_aspect('equal')
        plt.cla()
        plt.pcolormesh(X, Y, self.lattice, cmap=cm, shading='auto')
        plt.axis('off')
        plt.draw()
        if save:
            plt.savefig(f'plots/{name}.png')
        plt.pause(0.01)
        plt.cla()
    
    def measure(self):
        """
        Measures the energy and magnetisation of the system.

        Returns
        -------
        E : float
            Energy per spin of the system
        M : float
            Magnetisation per spin of the system
        """
        E, M = 0,0 

        config = self.lattice

        M =  np.sum(config)
        E = -np.sum(config*(np.roll(config, 1, axis=0)+
                            np.roll(config, 1, axis=1)))

        return E/self.N**2, abs(M)/self.N**2

    def change(self, pos):
        """
        Measures the change in energy and magnetisation when one spin is flipped.

        Parameters
        ----------
        pos : ndarray
            Location of the flipped spin

        Returns
        -------
        dE : float
            Change in energy of the lattice
        """
        [x,y] = pos

        dE = 2*self.lattice[y,x]*(self.lattice[(y-1)%self.N,x]+
                                         self.lattice[(y+1)%self.N,x]+
                                         self.lattice[y,(x-1)%self.N]+
                                         self.lattice[y,(x+1)%self.N])

        return dE

    def monte_carlo_step(self):
        """
        Performs the Metropolis algorithm to measure the average values for the observables.

        Parameters
        ----------
        steps : int
            Number of Monte Carlo steps for the algorithm to perform per temperature

        Returns
        -------
        E : float
            Average value of the energy per spin of the system
        M : float
            Average value of the magnetisation per spin of the system
        E2 : float
            Average value of the energy squared per spin of the system
        M2 : float
            Average value of the magnetisation squared per spin of the system
        """

        pos = np.random.randint(0, self.N, [2])

        dE = self.change(pos)

        if dE <= 0 or np.random.random() < np.exp((-1.0/self.T)*dE):
            self.lattice[pos[1], pos[0]] *= -1

    def monte_carlo_run(self, sweeps, sampleRate=1):
        """
        Runs a certain number of sweeps of the lattice and stores the energy and magnetisation.

        Parameters
        ----------
        sweeps : int
            number of sweeps of the lattice
        sampleRate : int, optional
            Frequency of recording the calculated value, by default 500

        Returns
        -------
        Ene, Mag : [tuple]
            Arrays of energy and magnetisation 
        """
        Mag = np.zeros(int((sweeps*self.N**2)/sampleRate)+1)
        Ene = np.zeros(int((sweeps*self.N**2)/sampleRate)+1)

        for i in range(int(self.f*sweeps*self.N**2)):
            self.monte_carlo_step()

            if self.live_show and i%2000 == 0:
                self.plot_lattice(thermalising=True)

        for i in range(int(sweeps*self.N**2)):
            self.monte_carlo_step()

            if i%sampleRate == 0:
                index = int(i/sampleRate)

                Ene[index], Mag[index] = self.measure()

            if self.live_show and i%2000 == 0:
                self.plot_lattice(thermalising=False)

        #self.plot_lattice(thermalising=False, name='snapshot_low_high', save=True)
            # if i==256*self.N**2:
            #     self.plot_lattice(thermalising=False, name=f'snapshot_i=256', save=True)
            # if i==4*self.N**2:
            #     self.plot_lattice(thermalising=False, name=f'snapshot_i={int(i/self.N**2)}', save=True)
            # if i==8*self.N**2:
            #     self.plot_lattice(thermalising=False, name=f'snapshot_i={int(i/self.N**2)}', save=True)
            # if i==16*self.N**2:
            #     self.plot_lattice(thermalising=False, name=f'snapshot_i={int(i/self.N**2)}', save=True)
            # if i==32*self.N**2:
            #     self.plot_lattice(thermalising=False, name=f'snapshot_i={int(i/self.N**2)}', save=True)
            # if i==64*self.N**2:
            #     self.plot_lattice(thermalising=False, name=f'snapshot_i={int(i/self.N**2)}', save=True)
            # if i==128*self.N**2:
            #     self.plot_lattice(thermalising=False, name=f'snapshot_i={int(i/self.N**2)}', save=True)
            # if i==256*self.N**2:
            #     self.plot_lattice(thermalising=False, name=f'snapshot_i={int(i/self.N**2)}', save=True)
            # if i==2*self.N**2:
            #     self.plot_lattice(thermalising=False, name=f'snapshot_i={int(i/self.N**2)}', save=True)
        return Ene, Mag
        
    def simulate(self, temperatures, sweeps):
        """
        Performs the Monte Carlo simulation over a range of temperatures.

        Parameters
        ----------
        temperatures : list
            Temperatures for the system to be simulated at
        sweeps : int
            Number of Monte Carlo sweeps of the lattice for the algorithm to perform per temperature

        Returns
        -------
        results : list
            List of tuples of the calculated values for each temperature
        """
        results=[]
        lattice = self.setup_lattice()

        for T in temperatures:
            self.T = T
            Ene, Mag = self.monte_carlo_run(sweeps)

            E = np.mean(Ene)
            E2 = np.mean(np.power(Ene, 2))

            M = np.mean(Mag)
            M2 = np.mean(np.power(Mag, 2))

            C = (E2 - E**2)/T**2
            X = (M2 - M**2)/T

            results.append((T, E, M, C, X))
            #print(f'T={T}, E={E}, M={M}, C={C}, X={X}')

        return results

    def plot_quantities(self, T_values):
        """
        Plots E, M, C, X against T

        Parameters
        ----------
        T_values : Tuple
            Data from the simulation
        """
        T, E, M, C, X = zip(*T_values)

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), 
                            constrained_layout=True)
        fig.suptitle(f'Ising Model size={self.N}')

        axs[0,0].plot(T, E)
        axs[0,0].grid()
        axs[0,0].set_ylabel('E')

        axs[0,1].plot(T, M)
        axs[0,1].grid()
        axs[0,1].set_ylabel('M')

        axs[1,0].plot(T, C)
        axs[1,0].grid()
        axs[1,0].set_ylabel('C')

        axs[1,1].plot(T, X)
        axs[1,1].grid()
        axs[1,1].set_ylabel('X')

        plt.show()

if __name__ == "__main__": 
    N=32
    T=1
    SWEEPS_ENE = 51
    SWEEPS_MAG = 101
    sweeps_ene_arr = np.arange(0, SWEEPS_ENE*N**2+1)/N**2
    sweeps_mag_arr = np.arange(0, SWEEPS_MAG*N**2+1)/N**2

    ising_ene = Ising(N=N, T=T, f=0, config='random')
    ene_ene, mag_ene = ising_ene.monte_carlo_run(sweeps=SWEEPS_ENE)

    ising_mag = Ising(N=N, T=T, f=0, config='random')
    ene_mag, mag_mag = ising_mag.monte_carlo_run(sweeps=SWEEPS_MAG)

    fig, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].plot(sweeps_ene_arr, ene_ene, color='k')
    axs[0].set_ylim(bottom=-2.1, top=0)
    axs[0].set_xlim(left=0, right=SWEEPS_ENE-1)
    axs[0].set_xlabel('Sweeps')
    axs[0].set_ylabel('$E$')

    axs[1].plot(sweeps_mag_arr, mag_mag, color='k')
    axs[1].set_ylim(bottom=0, top=1.1)
    axs[1].set_xlim(left=0, right=SWEEPS_MAG-1)
    axs[1].set_xlabel('Sweeps')
    axs[1].set_ylabel('$m$')

    plt.savefig('plots/energyevol.pdf')
    plt.show()
    