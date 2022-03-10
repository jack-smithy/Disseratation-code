import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
plt.style.use('science')
plt.rcParams['figure.figsize'] = 6,3
plt.rcParams['figure.constrained_layout.use'] = True

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

if __name__=='__main__':
    start = dt.datetime.now()
    Is = [216, 512]
    repeats = 10

    fig, axs = plt.subplots(nrows=1, ncols=2)

    for I in Is:
        N1min, N1max, step1 = int(0.05*I), int(0.22*I), 3
        N1s = np.arange(N1min, N1max, step1)

        N2min, N2max, step2 = int(0.11*I), int(0.17*I), 1
        N2s = np.arange(N2min, N2max, step2)

        N3s = np.append(N1s, N2s)

        res = []
        [res.append(x) for x in N3s if x not in res]
        Ns = np.sort(res)
        print(Ns)
 
        numNs = len(Ns)
        capacities = np.zeros(shape=(numNs, repeats), dtype=np.float64)
        capacities2 = np.zeros(shape=(numNs, repeats), dtype=np.float64)
        capacities4 = np.zeros(shape=(numNs, repeats), dtype=np.float64)

        energy = np.zeros(shape=(numNs, repeats), dtype=np.float64)
        energy2 = np.zeros(shape=(numNs, repeats), dtype=np.float64)

        for i in range(repeats):
            capacities_sync_arr = []
            energy_sync_arr = []
            print('----------------------------------------')
            print(f'Network size = {I}, run = {i+1}/{repeats} \n')
            for N in Ns:
                np.random.seed(i)
                model = Hopfield(I)
                data = generate_data(N, I)

                for j, item in enumerate(data):
                    model.make_weights(data[j])

                pattern = data[0]
                partial_pattern = np.where(pattern + np.random.normal(0,1, I) < 0.5, 0, 1)

                initial_overlap = model.overlap(partial_pattern, pattern)

                output_sync, e_list_sync = model.predict(partial_pattern, iter=200)

                final_energy_sync = e_list_sync[-1]
                final_overlap_sync = model.overlap(output_sync, pattern)

                capacities_sync_arr.append(final_overlap_sync)
                energy_sync_arr.append(final_energy_sync)

            capacities[:,i] = capacities_sync_arr
            capacities2[:,i] = np.power(capacities_sync_arr, 2)
            capacities4[:,i] = np.power(capacities_sync_arr, 4)

            energy[:,i] = energy_sync_arr
            energy2[:,i] = np.power(energy_sync_arr, 2)

        q = np.mean(capacities, axis=1)
        q2 = np.mean(capacities2, axis=1)
        q4 = np.mean(capacities4, axis=1)
        g = q4/q2**2

        e = np.mean(energy, axis=1)
        e2 = np.mean(energy2, axis=1)
        C = e2 - e**2

        print(dt.datetime.now()-start)

        # np.save(f'hopfield_data/q_I={I}', q)
        # np.save(f'hopfield_data/q2_I={I}', q2)
        # np.save(f'hopfield_data/q4_I={I}', q4)

        # np.save(f'hopfield_data/e_I={I}', e)
        # np.save(f'hopfield_data/e2_I={I}', e2)

        alpha = Ns/I
        axs[0].plot(alpha, q)
        axs[0].set_xlim((0.05, 0.2))

        axs[1].plot(alpha, C)
        axs[1].set_xlim((0.05, 0.2))
    plt.show()

