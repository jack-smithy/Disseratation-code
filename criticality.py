import numpy as np
import matplotlib.pyplot as plt
from hopfield import Hopfield, generate_data
plt.style.use('science')


Is = [512, 1024, 2048]
repeats = 20

for I in Is:
    Nmin, Nmax, step = int(0.05*I), int(0.22*I), int(0.005*I)
    Ns = np.arange(Nmin, Nmax, step)
    numNs = len(Ns)

    capacities = np.zeros(shape=(numNs, repeats), dtype=np.float64)
    capacities2 = np.zeros(shape=(numNs, repeats), dtype=np.float64)
    capacities4 = np.zeros(shape=(numNs, repeats), dtype=np.float64)

    for i in range(repeats):
        capacities_sync_arr = []
        print('----------------------------------------')
        print(f'Network size = {I}, run = {i+1}/{repeats} \n')
        for N in Ns:
            np.random.seed(i)
            model = Hopfield(I)
            data = generate_data(I, N)

            for j, item in enumerate(data):
                model.make_weights(data[j])

            pattern = data[0]
            partial_pattern = np.where(pattern + np.random.normal(0,1, I) < 0.5, 0, 1)

            initial_overlap = model.overlap(partial_pattern, pattern)

            output_sync, e_list_sync = model.predict(partial_pattern, pattern, iter=200)
            final_overlap_sync = model.overlap(output_sync, pattern)
            capacities_sync_arr.append(final_overlap_sync)

        capacities[:,i] = capacities_sync_arr
        capacities2[:,i] = np.power(capacities_sync_arr, 2)
        capacities4[:,i] = np.power(capacities_sync_arr, 4)

    q = np.mean(capacities, axis=1)
    q2 = np.mean(capacities2, axis=1)
    q4 = np.mean(capacities4, axis=1)

    np.save('hopfield_data/q', q)
    np.save('hopfield_data/q2', q2)
    np.save('hopfield_data/q4', q4)

    NIs = Ns/I
    g = q4/q2**2
    plt.plot(NIs, g, label=f'I={I}')

plt.xlim(0.05, 0.2)
#plt.ylim(0, 1.1)
plt.legend()
plt.xlabel('N/I')
plt.ylabel('Overlap')
plt.show()
