from timeit import repeat
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from numba import jit
#plt.style.use('science')
#plt.rcParams['figure.figsize'] = 6,3
#plt.rcParams['figure.constrained_layout.use'] = True

class Hopfield:
    def __init__(self, length):
        """
        Initialises the weights matrix with correct size populated with zeros

        Parameters
        ----------
        length : int
            Length of binary pattern for the network to store
        """
        self.length = length
        self.W = np.zeros([length, length])

    def make_weights(self, data:list) -> np.ndarray:
        """
        Checks the input is the correct size then updates the weights 
        matrix for one training example

        Parameters
        ----------
        data : list
            Pattern to update the weights matrix
        """

        N = len(data)
        if N != self.length:
            print(f"Data length is {N}, needs to be {self.length}")
            return 

        self.W += np.outer(data, data)
        for i in range(self.length):
            self.W[i,i] = 0

    def update_neuron(self, state, index=None):
        """
        Updates a given state by multiplying by the weights matrix and passing throygh activation function

        Parameters
        ----------
        state : list
            Input pattern
        index : int, optional
            Index of element to update for asynchronous updating, by default None

        Returns
        -------
        state : list
            State after one update iteration
        """
        if index == None:
            new_state = np.matmul(self.W, state)
            new_state[new_state < 0] = -1
            new_state[new_state > 0] = 1
            new_state[new_state == 0] = state[new_state == 0]
            state = new_state

        else:
            a = np.matmul(self.W[index], state)
            if a < 0:
                state[index] = -1
            elif a > 0:
                state[index] = 1

        return state

    def predict(self, input, iter, asyn=False, asyn_iter = 200):
        """
        Iterates the update procedure

        Parameters
        ----------
        input : list
            
        pattern : list
            Stored pattern the network is trying to converge to
        iter : int
            iterations
        asyn : bool, optional
            Whether to iterate synchronously or asynchronously, by default False
        asyn_iter : int, optional
            asynchronous iterations, by default 200

        Returns
        -------
        state : list
            Final state prediction

        e_list : list
            List of energies for each iteration

        overlap_list : list
            List of overlap between final state prediction and stored memory
        """
        e_list = []
        e = self.energy(input)
        e_list.append(e)
        state = input

        # if asyn == True:
        #     for i in range(iter):
        #         for j in range(asyn_iter):
        #             index = np.random.randint(input_length)
        #             state = self.update_neuron(state, index)
        #         overlap = self.overlap(state, pattern)
        #         new_e = -0.5*np.matmul(np.matmul(np.transpose(state), self.W) ,state)
        #         new_e = self.energy(state)
        #         #if i%10==0:
        #         #    print(f'Iteration number {i}, Energy={new_e}')
        #         if new_e == e:
        #             #print("\nEnergy unchanged, updates stop")
        #             #print(f'\nNumber of asynchronous iterations={i}')
        #             break
        #         e = new_e
        #         e_list.append(e)

        if asyn == False:
            for i in range(iter):
                state = self.update_neuron(state)
                new_e = self.energy(state)
                if new_e == e:
                    break
                e = new_e
                e_list.append(e)
    
        return state, e_list

    def energy(self, vec):
        """
        Calculates energy for a given state

        Parameters
        ----------
        vec : list
            State which the energy is calculated for

        Returns
        -------
        e : float
            Energy of the given state
        """
        e = -0.5*np.matmul(np.matmul(np.transpose(vec), self.W), vec)
        return e

    def overlap(self, u, v):
        """
        Overlap between two states. Measure of similarity through scaled dot product

        Parameters
        ----------
        u, v : tuple
            States to be compared

        Returns
        -------
        overlap : float
            Similarity between two inputs
        """
        if len(u) != len(v):
            print("Vectors not same length")
            return

        overlap = abs(np.dot(u, v)/self.length)
        return overlap

def generate_data(num, length):
    """
    Generates random data of correct length

    Parameters
    ----------
    length : int
        Length of pattern
    num : int
        Number of patterns

    Returns
    -------
    data : ndarray
        Array of randomly generated data
    """
    data = np.zeros(shape=(num, length))
    for n in range(int(num)):
        data[n,:] = np.random.choice([-1,1], size=length) 
    return data.astype(int)

def test_capacity(alpha, N):
    print('----------------------------------------')
    print(f'Size = {N}, Testing capacity alpha = {alpha} \n')

    P = int(N*alpha)

    model = Hopfield(N)
    data = generate_data(P, N)

    for j in range(P):
        model.make_weights(data[j,:])

    q_mu_p = []
    for j in range(P):
        pattern = data[j,:]
        partial_pattern = np.where(pattern + np.random.normal(0,1, N) < 0, -1, 1)
        output, e_list = model.predict(partial_pattern, iter=100)
        overlap = model.overlap(output, pattern)
        q_mu_p.append(overlap)

    np.mean(q_mu_p)

    return np.mean(q_mu_p)

if __name__=='__main__':
    numAlphas = 11
    alphas = np.linspace(0.04, 0.32, numAlphas)

    repeats = 3
    N = 256

    test_capacity_vec = np.vectorize(test_capacity)
    q_avg = np.zeros(shape=(repeats, numAlphas))
    for i in range(repeats):
        q_avg[i,:] = test_capacity_vec(alphas, N)
    
    np.save(f'hopfield_data/qvals_N={N}_run_1.npy', q_avg)

    # q_avg_2 = np.zeros(shape=(repeats, numAlphas))
    # for i in range(repeats):
    #     q_avg_2[i,:] = test_capacity_vec(alphas, N)

    # q_avg = np.append(q_avg_2, q_avg_2, axis=0)    
    # q = np.mean(q_avg, axis=0)
    
    # plt.plot(alphas, q)
    # plt.show()