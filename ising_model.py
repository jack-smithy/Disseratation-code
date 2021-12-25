import numpy as np
import matplotlib.pyplot as plt
import random

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

        norm = float(self.N**2)

        return Esum , Msum, E2sum, M2sum


    def simulate(self, temperatures, steps):
        results=[]
        #self.live_show = True
        lattice = self.setup_lattice()
        size = self.N**2

        for T in temperatures:
            self.T = T
            E, M, E2, M2 = self.monte_carlo(steps)

            E = E/(size*steps)
            M = M/(size*steps)
            C = (E2/(size*steps)-E**2/(size*steps**2))/T**2
            X = (M2/(size*steps)-M**2/(size*steps**2))/T

            results.append((T, E, M, C, X))
            print(f'T={T}, E={E}, M={M}, C={C}, X={X}')
        return results


    def plot(self, T_values):
        plt.close('all')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ioff()
        T, E, absM, C, X = zip(*T_values)

        fig, axs = plt.subplots(2,2, figsize=(6,4), constrained_layout=True)
        fig.suptitle(f'Ising Model size={self.N}')

        axs[0,0].plot(T, absM)
        axs[0,0].set_ylabel('M')

        axs[0,1].plot(T, E)
        axs[0,1].set_ylabel('E')

        axs[1,0].plot(T, C)
        axs[1,0].set_ylabel('C')

        axs[1,1].plot(T, X)
        axs[1,1].set_ylabel('X')

        #plt.savefig(f'ising_size={self.N}')
        plt.show()

    def c_for_size(self, T_values):
        plt.close('all')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ioff()
        T, E, absM, C, X = zip(*T_values)

        return T, E, absM, C, X


if __name__=='__main__':
    sizes = [32, 64, 96, 128]
    C_vals = []
    X_vals = []
    temperatures = np.linspace(1,4, 31)

    for size in sizes:
        s = Ising(N=size, live_show=True)
        results = s.simulate(temperatures, 50000)
        plt.close('all')
        T, E, absM, C, X = s.c_for_size(results)
        C_vals.append(C)
        X_vals.append(X)

    fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols= 2, figsize=(8,4))

    plt.style.use('classic')

    ax0.set_title("C")
    ax0.grid(True)
    ax0.plot(temperatures, C_vals[0])
    ax0.scatter(temperatures ,C_vals[0], s=10, marker='v')
    ax0.plot(temperatures, C_vals[1])
    ax0.scatter(temperatures ,C_vals[1], s=10, marker='o')
    ax0.plot(temperatures, C_vals[2])
    ax0.scatter(temperatures ,C_vals[2], s=10, marker='P')
    ax0.plot(temperatures, C_vals[3])
    ax0.scatter(temperatures ,C_vals[3], s=10, marker='D')

    ax1.set_title("X")
    ax1.grid(True)
    ax1.plot(temperatures, X_vals[0])
    ax1.scatter(temperatures ,X_vals[0], s=10, marker='v')
    ax1.plot(temperatures, X_vals[1])
    ax1.scatter(temperatures ,X_vals[1], s=10, marker='o')
    ax1.plot(temperatures, X_vals[2])
    ax1.scatter(temperatures ,X_vals[2], s=10, marker='+')
    ax1.plot(temperatures, X_vals[3])
    ax1.scatter(temperatures ,X_vals[3], s=10, marker='D')

    plt.show()
    
