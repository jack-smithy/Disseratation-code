from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
plt.style.use('classic')


class Hopfield:
    def __init__(self, length):
        self.length = length
        self.W = np.zeros([length, length])

    def make_weights(self, data):
        N = len(data)

        if N != self.length:
            print(f"Data length is {N}, needs to be {self.length}")
            return 
        
        for i in range(self.length):
            self.W[i,:] += (data[i]/N)*data
            self.W[i,i] = 0


    def update_neuron(self, state, index=None):
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
                if i%10==0:
                    print(f'Iteration number {i}, Energy={new_e}')
                if new_e == e:
                    print("\nEnergy unchanged, updates stop")
                    print(f'\nNumber of asynchronous iterations={i}')
                    break
                e = new_e
                e_list.append(e)

        else:
            for i in range(iter):
                state = self.update_neuron(state)
                new_e = self.energy(state)
                overlap = self.overlap(state, pattern)
                overlap_list.append(overlap)
                if i%10==0:
                    print(f'Iteration number {i}, Energy={new_e}')
                if new_e == e:
                    print("\nEnergy unchanged, updates stop")
                    print(f'\nNumber of synchronous iterations={i}')
                    break
                e = new_e
                e_list.append(e)
        return state, e_list, overlap_list

    def energy(self, vec):
        e = -0.5*np.matmul(np.matmul(np.transpose(vec), self.W), vec)
        return e

    def overlap(self, u, v):
        if len(u) != len(v):
            print("Vectors not same length")
            return

        return abs(np.dot(u, v)/self.length)

def generate_data(length, num):
    data = []
    for n in range(num):
        pattern = np.random.choice([-1, 1], size=length)
        data.append(pattern) 

    return data
        

if __name__=="__main__":

    I = 128
    N = 5

    model = Hopfield(I)
    data = generate_data(I, N)

    for i, item in enumerate(data):
        print(f'Training pattern {i+1}')
        model.make_weights(data[i])
    print('----------------------------------------')

    pattern = data[0]
    partial_pattern = np.where(pattern + np.random.normal(-1,1, I) < 0.5, 0, 1)

    initial_overlap = model.overlap(partial_pattern, pattern)
    print(f'Initial overlap = {initial_overlap} \n')

    output_async, e_list_async, overlap_list_asyn = model.predict(partial_pattern, pattern, 100, asyn=True)
    print('')
    output_sync, e_list_sync, overlap_list_syn = model.predict(partial_pattern, pattern, 100, asyn=False)

    final_overlap_asyn = model.overlap(output_async, pattern)
    final_overlap_syn = model.overlap(output_sync, pattern)

    print('')
    print('----------------------------------------')
    print(f'Final overlap asynchronous updates = {final_overlap_asyn}')
    print(f'Final overlap synchronous updates = {final_overlap_syn}')

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].set_title('Energy')
    axs[0].plot(e_list_sync)
    axs[0].plot(e_list_async)

    axs[1].set_title('Overlap')
    axs[1].plot(overlap_list_syn)
    axs[1].plot(overlap_list_asyn)
    plt.show()
    
    

    

    
    
    
