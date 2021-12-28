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
            C = (1/steps**2)*(E2/(size)-E**2/(size*steps))/T**2
            X = (1/steps**2)*(M2/(size)-M**2/(size*steps))/T

            results.append((T, E, M, C, X))
            print(f'T={T}, E={E}, M={M}, C={C}, X={X}')

        return results

    def make_tuple(self, T_values):
        plt.close('all')
        plt.ioff()
        T, E, absM, C, X = zip(*T_values)

        return T, E, absM, C, X

'''
if __name__=='__main__':
    s = Ising(N=64, live_show=True)
    results = s.simulate(np.linspace(0.1, 6, 40), 100000)
    T, E, M, C, X = s.make_tuple(results)

    plt.plot(T, C)
    plt.show()

'''



