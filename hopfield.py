import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
plt.style.use('science')


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

    def make_weights(self, data):
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
        
        for i in range(self.length):
            self.W[i,:] += (data[i]/N)*data
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

    def predict(self, input, pattern, iter, asyn=False, asyn_iter = 200):
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
        input_length = len(input)
        e_list = []
        overlap_list = []
        e = self.energy(input)
        e_list.append(e)
        state = input

        if asyn:
            for i in range(iter):
                for j in range(asyn_iter):
                    index = np.random.randint(input_length)
                    state = self.update_neuron(state, index)
                    overlap = self.overlap(state, pattern)
                    overlap_list.append(overlap)
                new_e = -0.5*np.matmul(np.matmul(np.transpose(state), self.W) ,state)
                new_e = self.energy(state)
                #if i%10==0:
                #    print(f'Iteration number {i}, Energy={new_e}')
                if new_e == e:
                    #print("\nEnergy unchanged, updates stop")
                    #print(f'\nNumber of asynchronous iterations={i}')
                    break
                e = new_e
                e_list.append(e)

        else:
            for i in range(iter):
                state = self.update_neuron(state)
                new_e = self.energy(state)
                overlap = self.overlap(state, pattern)
                overlap_list.append(overlap)
                #if i%10==0:
                #    print(f'Iteration number {i}, Energy={new_e}')
                if new_e == e:
                    #print("\nEnergy unchanged, updates stop")
                    #print(f'\nNumber of synchronous iterations={i}')
                    break
                e = new_e
                e_list.append(e)
    
        return state, e_list, overlap_list

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

def generate_data(length, num):
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
    data = []
    for n in range(int(num)):
        pattern = np.random.choice([-1, 1], size=length)
        data.append(pattern) 

    return data
    

if __name__=="__main__":
    Is = [4000, 6000, 8000, 1000]
    np.random.seed(1)

    fig, axs = plt.subplots(nrows=1, ncols=1)

    for I in Is:
        Nmin, Nmax = 0.05*I, 0.25*I
        step = 5
        Ns = np.arange(Nmin, Nmax, step)
        capacities_sync = []
        capacities_async = []

        for N in Ns:
            print('----------------------------------------')
            print(f'Testing capacity = {N}')
            model = Hopfield(I)
            data = generate_data(I, N)

            for i, item in enumerate(data):
                #if i%20 == 0:
                #    print(f'Training pattern {i}')
                model.make_weights(data[i])

            pattern = data[0]
            partial_pattern = np.where(pattern + np.random.normal(0,1, I) < 0.5, 0, 1)

            initial_overlap = model.overlap(partial_pattern, pattern)
            print(f'Initial overlap = {initial_overlap} \n')

            #output_async, e_list_async, overlap_list_async = model.predict(partial_pattern, pattern, iter=200, asyn=True)
            output_sync, e_list_sync, overlap_list_sync = model.predict(partial_pattern, pattern, iter=200)

            #final_overlap_asyn = model.overlap(output_async, pattern)
            final_overlap_syn = model.overlap(output_sync, pattern)

            #capacities_async.append(final_overlap_asyn)
            capacities_sync.append(final_overlap_syn)

            #print(f'Final overlap asynchronous updates = {final_overlap_asyn}')
            print(f'Final overlap synchronous updates = {final_overlap_syn}')
            print('----------------------------------------')
            print('')

            NIs = Ns/I

    axs.plot(NIs, capacities_sync)
    #axs[1].plot(NIs, capacities_sync)

    axs.set_ylim((0, 1.1))
    #axs[1].set_ylim((0, 1.1))
    axs.set_xlim((0.05, 0.2))
    #axs[1].set_xlim((0.05, 0.2))
    axs.set_ylabel('Overlap')
    axs.set_xlabel('N/I')
    #axs[1].set_xlabel('N/I')
    #plt.legend()
    plt.show()

    


    
    

    

    
    
    
