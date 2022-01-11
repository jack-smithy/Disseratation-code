import numpy as np
import matplotlib.pyplot as plt
from ising_model import Ising

class Cumulant(Ising):
    def __init__(self, N, J=1, T=1, f=0.2, live_show=True):
        super().__init__(N, J=J, T=T, f=f, live_show=True)

    def monte_carlo_cumulant(self, steps):
        """
        Performs the Metropolis algorithm to measure the average values for the observables.

        Parameters
        ----------
        steps : int
            Number of Monte Carlo steps for the algorithm to perform per temperature

        Returns
        -------
        M2 : float
            Average value of the magnetisation squared per spin of the system
        M4 : float
            Average value of the magnetisation^4 per spin of the system
        """
        E, M = self.measure()
        M2, M4 = M**2, M**4

        M2sum = M2
        M4sum = M4

        for n in range(int((1+self.f)*steps)):
            pos = np.random.randint(0, self.N, [2])

            dE, dM = self.change(pos)

            if dE <= 0 or np.random.random() < np.exp((-1.0/self.T)*dE):
                self.lattice[pos[1], pos[0]] *= -1

                M += dM
                M2 += M**2
                M4 += M**4
            
            if n>self.f*steps:
                M2sum += M2
                M4sum += M4

                thermalising = False

            if n<=self.f*steps:
                thermalising = True

            if self.live_show and n%2000==0:
                self.plot_lattice(thermalising)

        norm = steps*self.N**2

        M2 = M2sum/norm
        M4 = M4sum/norm

        return M2, M4

    def simulate_cumulant(self, temperatures, steps):
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
            List of tuples of the temperature and binder cumulant value for each temperature
            
        """
        results=[]
        lattice = self.setup_lattice()

        for T in temperatures:
            self.T = T
            M2, M4 = self.monte_carlo_cumulant(steps)

            g = 0.5 * (3- M4/(M2**2))/self.N

            results.append((T, g))
            print(f'T={T}, g={g}')

        return results

    def get_cumulant_values(self, T_values):
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
            The binder cumulant for each temperature
        """
        plt.close('all')
        plt.ioff()
        T, g = zip(*T_values)

        return T, g

    def plot_cumulant(self, T_values):
        T, g = self.get_cumulant_values(T_values)
        plt.plot(T, g)
        plt.show()





if __name__ == "__main__":
    sizes = [64, 72, 96]
    temps = np.linspace(2, 3, 6)
    g_vals = []
    for size in sizes:
        g = Cumulant(N=size)
        results = g.simulate_cumulant(temps, 10000)
        T, g = g.get_cumulant_values(results)
        g_vals.append(g)

    plt.plot(T, g_vals[0])
    plt.plot(T, g_vals[1])
    plt.plot(T, g_vals[2])
    plt.show()

        


