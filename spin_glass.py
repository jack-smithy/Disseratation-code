import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
plt.style.use('classic')

class Spin(object):
    def __init__(self, N, T=1.0, f=0.2, live_show=False):
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

        self.setup_lattice()
        self.interaction_weights()

        if live_show == True:
            self.setup_plotting()

    def setup_lattice(self):
        """
        Setup the lattice with a configuration of all up spins.
        """
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

    def plot_lattice(self, thermalising):
        """
        Plot the lattice spins on an NxN grid.

        Parameters
        ----------
        thermalising : bool
            Shows the lattice in a different colour when the system is thermalising
        """
        X, Y = np.meshgrid(range(self.N), range(self.N))
        cm = plt.cm.Blues

        if thermalising:
            cm = plt.cm.Greys
        
        plt.cla()
        plt.pcolormesh(X, Y, self.lattice, cmap=cm, shading='auto')
        plt.axis('off')
        plt.draw()
        plt.pause(0.01)
        plt.cla()

    def interaction_weights(self, dist='Normal'):
        if dist == 'Normal':
            self.J = np.random.normal(0, 1/self.N, size=[self.N, self.N])
            self.J = 0.5*(self.J + np.transpose(self.J))
            for i in range(self.N):
                self.J[i,i] = 0

        elif dist == 'Bimodal':
            self.J = np.random.choice([-1, 1], size=[self.N, self.N])
            self.J = 0.5*(self.J + np.transpose(self.J))
            for i in range(self.N):
                self.J[i,i] = 0
        
        else:
            print('Not recognised distribution: Must be "Normal" or "Bimodal"')
            pass

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
        E, M = 0, 0
        for y in range(self.N):
            for x in range(self.N):
                E += self.J[y,x]*self.lattice[y,x]*(self.lattice[(y-1)%self.N,x] + self.lattice[y,(x-1)%self.N])
                M += self.J[y,x]*self.lattice[y,x]

        return -1*E/self.N**2, abs(M)/self.N**2

    def change(self, pos):
        """
        Measures the change in energy when one spin is flipped.

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

        dE = 2*self.J[y,x]*self.lattice[y,x]*(self.lattice[(y-1)%self.N,x]+
                                              self.lattice[(y+1)%self.N,x]+
                                              self.lattice[y,(x-1)%self.N]+
                                              self.lattice[y,(x+1)%self.N])

        return dE

    def monte_carlo_step(self):
        pos = np.random.randint(0, self.N, [2])

        dE = self.change(pos)

        if dE <= 0 or np.random.random() < np.exp((-1.0/self.T)*dE):
            self.lattice[pos[1], pos[0]] *= -1

    def monte_carlo_run(self, sweeps, sampleRate=200):
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

        return Ene, Mag

    def q(self, S):
        Si = S.flatten()
        return np.dot(self.lattice, Si)/self.N**2
        
if __name__ == "__main__":

    '''
    Ts = np.linspace(0.5, 5.5, 21)
    Sweeps = 500
    Ms = []
    for T in Ts:
        print(f'On T={T}')
        spin = Spin(N=16, T=T)
        Ene, Mag = spin.monte_carlo_run(sweeps=Sweeps)
        M = np.mean(Mag)
        Ms.append(M)
        print(f'M={M}')

    plt.plot(Ts, Ms)
    plt.show()
    '''





