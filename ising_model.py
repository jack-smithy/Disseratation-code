import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

class Ising(object):
    def __init__(self, N, J=1.0, T=1, f=0.2, live_show=False):
        '''
        param N: Lattice size NxN
        param J: Interaction strength mean
        param T: Temperature of the system
        param f: Thermalising steps
        '''

        self.N=N
        self.J=J
        self.T=T
        self.f=f
        self.live_show = live_show

        self.setup_lattice()
        self.setup_plotting()

    def setup_lattice(self):
        '''Randomly initialise spins'''
        self.lattice = np.ones([self.N, self.N])

    def setup_plotting(self):
        plt.close('all')
        self.live = plt.figure(figsize=(4, 4))
        plt.ion()
        plt.axis('off')
        plt.show()

    def plot_lattice(self, thermalising=False):
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
        E, M = 0,0 

        config = self.lattice

        M =  np.sum(config)
        E = -np.sum(config*(np.roll(config, 1, axis=0)+
                            np.roll(config, 1, axis=1)))

        return E, M

    def change(self, pos):

        [x,y] = pos

        dE = 2*self.J*self.lattice[y,x]*(self.lattice[(y-1)%self.N,x]+
                                         self.lattice[(y+1)%self.N,x]+
                                         self.lattice[y,(x-1)%self.N]+
                                         self.lattice[y,(x+1)%self.N])

        dM = -2*self.lattice[y,x]

        return dE, dM

    def monte_carlo(self, steps):
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

        return Esum, Msum, E2sum, M2sum

    def simulate(self, temperatures, steps):
        results=[]
        lattice = self.setup_lattice()
        size = self.N**2

        for T in temperatures:
            self.T = T
            Esum, Msum, E2sum, M2sum = self.monte_carlo(steps)

            E = Esum/(size*steps)
            M = Msum/(size*steps)

            C = (E2sum/(size*steps) - (E/(size*steps))**2)/(steps*T**2)
            X = (M2sum - M**2)/T

            results.append((T, E, M, C, X))
            print(f'T={T}, E={E}, M={M}, C={C}, X={X}')

        return results

    def make_tuple(self, T_values):
        plt.close('all')
        plt.ioff()
        T, E, absM, C, X = zip(*T_values)

        return T, E, absM, C, X

    def plot_quantities(self, T_values):

        T, E, M, C, X = self.make_tuple(T_values)

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


s = Ising(N=64, live_show=True)
results = s.simulate(np.linspace(1,4,11), 50000)
s.plot_quantities(results)






