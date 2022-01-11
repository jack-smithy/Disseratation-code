import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

class Ising(object):
    def __init__(self, N, J=1.0, T=1.0, f=0.2, live_show=False):
        """
        Initialises the lattice with the necessary attributes.

        Parameters
        ----------
        N : int
            Size of the lattice side
        J : float, optional
            Interaction strength between spins, by default 1.0
        T : float, optional
            Temparature of the system, by default 1J
        f : float, optional
            [description], by default 0.2
        live_show : bool
            Show a live snapshot of the lattice, by default False
        """
        self.N=N
        self.J=J
        self.T=T
        self.f=f
        self.live_show = live_show

        self.setup_lattice()
        self.setup_plotting()

    def setup_lattice(self):
        """
        Sets up NxN lattice with randomly initialised spins.
        """
        self.lattice = np.ones([self.N, self.N])

    def setup_plotting(self):
        """
        Creates axes for live plotting
        """
        plt.close('all')
        self.live = plt.figure(figsize=(4, 4))
        plt.ion()
        plt.axis('off')
        plt.show()

    def plot_lattice(self, thermalising):
        """
        Show the lattice spins on an NxN grid.

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
    
    def measure(self):
        """
        Measures the energy and magnetisation of the system.

        Returns
        -------
        E : float
            Energy of the system
        M : float
            Magnetisation of the system
        """
        E, M = 0,0 

        config = self.lattice

        M =  np.sum(config)
        E = -np.sum(config*(np.roll(config, 1, axis=0)+
                            np.roll(config, 1, axis=1)))

        return E, M

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
        dM : float
            Change in magnetisation of the lattice
        """
        [x,y] = pos

        dE = 2*self.J*self.lattice[y,x]*(self.lattice[(y-1)%self.N,x]+
                                         self.lattice[(y+1)%self.N,x]+
                                         self.lattice[y,(x-1)%self.N]+
                                         self.lattice[y,(x+1)%self.N])

        dM = -2*self.lattice[y,x]

        return dE, dM

    def monte_carlo(self, steps):
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
        E, M = self.measure()
        E2, M2 = E**2, M**2
  
        Esum = E
        Msum = abs(M)
        E2sum = E2
        M2sum = M2

        for n in range(int((1+self.f)*steps)):
            pos = np.random.randint(0, self.N, [2])

            dE, dM = self.change(pos)

            if dE <= 0 or np.random.random() < np.exp((-1.0/self.T)*dE):
                self.lattice[pos[1], pos[0]] *= -1

                E += dE
                M += dM

                E2 += E**2
                M2 += M**2
            
            if n>self.f*steps:
                Esum += E
                Msum += abs(M)
                E2sum += E2
                M2sum += M2

                thermalising = False

            if n<=self.f*steps:
                thermalising = True

            if self.live_show and n%2000==0:
                self.plot_lattice(thermalising)

        norm = steps*self.N**2

        return Esum/norm , Msum/norm , E2sum/norm , M2sum/norm

    def simulate(self, temperatures, steps):
        """
        Performs the Monte Carlo simulation over a range of temperatures.

        Parameters
        ----------
        temperatures : list
            Temperatures for the system to be simulated at
        steps : int
            Number of Monte Carlo steps for the algorithm to perform per temperature

        Returns
        -------
        results : list
            List of tuples of the calculated values for each temperature
            
        """
        results=[]
        lattice = self.setup_lattice()

        for T in temperatures:
            self.T = T
            E, M, E2, M2 = self.monte_carlo(steps)

            C = (E2 - E**2)/(steps*T**2)
            X = (M2 - M**2)/(steps*T)

            results.append((T, E, M, C, X))
            print(f'T={T}, E={E}, M={M}, C={C}, X={X}')

            return results

    def get_values(self, T_values):
        """
        Changes a lift of tuples into a tuple of lists

        Parameters
        ----------
        T_values : list
            The results returned by the simulate function

        Returns
        -------
        T : list
            The temperatures of each simulation
        E : list
            The energy for each temperature
        M : list
            The magnetisation for each temperature
        C : list
            The specific heat for each temperature
        X : list
            The magnetic susceptibility for each temperature
        """
        plt.close('all')
        plt.ioff()
        T, E, absM, C, X = zip(*T_values)

        return T, E, absM, C, X

    def plot_quantities(self, T_values):
        """
        plot_quantities [summary]

        Parameters
        ----------
        T_values : [type]
            [description]
        """
        T, E, M, C, X = self.get_values(T_values)

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6,4), 
                            constrained_layout=True)
        fig.suptitle(f'Ising Model size={self.N}')

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


if __name__ == "__main__":
    start = dt.datetime.now()
    s = Ising(N=16, live_show=False)
    results = s.simulate(np.linspace(1,4,6), 50000)
    print(dt.datetime.now()-start)
    s.plot_quantities(results)
