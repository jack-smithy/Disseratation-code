import numpy as np
import matplotlib.pyplot as plt

class Ising_sim():
    def __init__(self, N, J=1.0, h=0, T=1):
        '''
        param N: Lattice size NxN
        param J: Interaction strength
        param h: External magnetic field strength
        param T: Temperature of the system
        '''

        self.N=N
        self.J=J
        self.h=h
        self.T=T

        self.setup_lattice()
        self.setup_plotting()

        self.live_show = True

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
        cm = plt.cm.RdBu
        if thermalising:
            cm = plt.cm.Greys
        plt.cla()
        plt.pcolormesh(X, Y, self.lattice, cmap=cm, shading='auto')
        plt.axis('off')
        plt.draw()
        plt.pause(0.01)
        plt.cla()
    
    def measure(self):
        E, M, Esq, Msq = 0,0,0,0

        for x in range(self.N):
            for y in range(self.N):
                E += self.lattice[x,y]*(self.lattice[(x-1)%self.N,y]+
                                         self.lattice[x,(y-1)%self.N])

                Esq += (self.lattice[x,y]*(self.lattice[(x-1)%self.N,y]+
                                         self.lattice[x,(y-1)%self.N]))**2
                M += self.lattice[x,y]

                Msq += (self.lattice[x,y])**2
        
        E*=-1
        
        return E, M, Esq, Msq

    def change(self, pos):
        [x,y]=pos
        dE = -2*self.lattice[x,y]*(self.lattice[(x+1)%self.N,y]+
                                   self.lattice[(x-1)%self.N,y]+
                                   self.lattice[x,(y+1)%self.N]+
                                   self.lattice[x,(y-1)%self.N])
        dM = -2*self.lattice[x,y]
        dE = -1*self.J*dE-self.h*dM

        return dE, dM

    def monte_carlo(self, steps):
        E, M, Esq, Msq = self.measure()
        norm = float(steps*self.N**2)
  
        Esum = E
        absMsum = abs(M)
        Esqsum = Esq
        Msqsum = Msq

        for n in range(int(steps)):
            init_x = np.random.randint(0, self.N)
            init_y = np.random.randint(0, self.N)
            pos = [init_x, init_y]
            (dE, dM) = self.change(pos)

            if dE <= 0 or np.random.random() < np.exp((-1.0/self.T)*dE):
                self.lattice[pos[0], pos[1]] *= -1
                E += dE
                M += dM
                Esq += dE**2
                Msq += dM**2
            
            if n>0:
                Esum += E
                absMsum += abs(M)
                Esqsum += Esq
                Msqsum += Msq

            if self.live_show and n%2000==0:
                self.plot_lattice()

    
        return Esum/norm, absMsum/norm, Esqsum/norm, Msqsum/norm


    def simulate(self, temperatures, steps):
        results=[]
        self.live_show = True
        lattice = self.setup_lattice()
        for T in temperatures:
            self.T = T
            E, M, Esq, Msq = self.monte_carlo(steps)
            C = (Esq - E**2)/T**2
            X = (Msq - M**2)/T
            results.append((T, E, M, C, X))
            print(T, E, M, C, X)
        return results


    def plot_magnetisation(self, T_E_M_C_X_values):
        plt.close('all')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ioff()
        T, E, absM, C, X = zip(*T_E_M_C_X_values)

        fig, axs = plt.subplots(4, sharex=True)
        fig.suptitle('Ising Model')

        axs[0].plot(T, absM)
        axs[0].set_ylabel('M')

        axs[1].plot(T, E)
        axs[1].set_ylabel('E')

        axs[2].plot(T, C)
        axs[2].set_ylabel('C')

        axs[3].plot(T, X)
        axs[3].set_ylabel('X')
        axs[3].set_xlabel('Temperature')

        plt.show()

if __name__=='__main__':
    s = Ising_sim(40)
    results = s.simulate(np.linspace(1.5, 3, 30), 50000)
    s.plot_magnetisation(results)
