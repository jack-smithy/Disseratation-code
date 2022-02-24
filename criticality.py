import numpy as np
import matplotlib.pyplot as plt
from hopfield import Hopfield, generate_data
plt.style.use('science')

Is = [1000, 2000, 3000]
np.random.seed(1)

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7,3))

for I in Is:
    Nmin, Nmax, step = int(0.05*I), int(0.25*I), 5
    Ns = np.arange(Nmin, Nmax, step)
    capacities_sync = []
    capacities_async = []

    for N in Ns:
        print('----------------------------------------')
        print(f'Network size = {I}, Testing capacity = {N} \n')
        model = Hopfield(I)
        data = generate_data(I, N)

        for i, item in enumerate(data):
            model.make_weights(data[i])

        pattern = data[0]
        partial_pattern = np.where(pattern + np.random.normal(0,1, I) < 0.5, 0, 1)

        initial_overlap = model.overlap(partial_pattern, pattern)

        output_async, e_list_async, overlap_list_async = model.predict(partial_pattern, pattern, iter=200, asyn=True)
        output_sync, e_list_sync, overlap_list_sync = model.predict(partial_pattern, pattern, iter=200)

        final_overlap_asyn = model.overlap(output_async, pattern)
        final_overlap_syn = model.overlap(output_sync, pattern)

        capacities_async.append(final_overlap_asyn)
        capacities_sync.append(final_overlap_syn)

    NIs = Ns/I
    axs[0].plot(NIs, capacities_sync, label = f'I={I}')
    axs[1].plot(NIs, capacities_async, label = f'I={I}')

axs[0].set_xlim((0.05, 0.2))
axs[0].set_xlabel('N/I')

axs[0].set_ylim((0, 1.1))
axs[0].set_ylabel('Overlap')

axs[1].set_xlim((0.05, 0.2))
axs[1].set_xlabel('N/I')

plt.legend()
plt.show()