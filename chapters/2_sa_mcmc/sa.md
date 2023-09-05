# Simulated Annealing

## Map MAX-CUT problem to Ising Model

When we map various problems to the Ising model, the first step is to choose an appropriate energy function so that the ground state of the Ising Hamiltonian can represent the optimal solution of the problem.

For this `MAX-CUT` problem, we will configure the parameters $J_{ij}$ as the negative of edge weight, $-W_{ij}$ since

$$H = -J_{ij}\sum_{(i < j)} \sigma_i \sigma_j =  \sum_{\sigma_i=-\sigma_j}W_{ij} \sigma_i \sigma_j +  \sum_{\sigma_i=\sigma_j}W_{ij} \sigma_i \sigma_j = \sum_{\sigma_i=-\sigma_j}W_{ij}  -  \sum_{\sigma_i=\sigma_j}W_{ij}$$

For $W_{ij}$, we have

$$\sum W_{ij} =\sum_{\sigma_i=\sigma_j}W_{ij} + \sum_{\sigma_i=-\sigma_j}W_{ij} $$

Finally, we have,

$$ H = -2\sum_{\sigma_i=-\sigma_j}W_{ij} + \sum W_{ij} $$

For each `MAX-CUT` problem, $\sum W_{ij}$ is a constant. Therefore we showed that the ground state of the Ising Hamiltonian can represent the optimal solution of the problem, since we maximized

$$\sum_{\sigma_i=-\sigma_j}W_{ij}$$

````{admonition} Python implementation
:class: dropdown
```python
import numpy as np
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc

temps = []
bests = []
eners = []
# def ising_energy2(spins, weights):
#     energy = -0.5 * np.sum(weights * (1 - np.outer(spins, spins)))
#     return energy

max_cut_ising_energy = lambda spins, weights: sum([weights[i,j] * spins[i] * spins[j] for i in range(len(spins)) for j in range(i + 1, len(spins))])
def simulated_annealing(spins, weights, initial_temp, final_temp, cooling_rate, num_steps):
    current_spins = spins.copy()
    current_energy = max_cut_ising_energy(current_spins, weights)
    curr_step = 0
    best_spins = current_spins.copy()
    best_energy = current_energy

    temp = initial_temp

    while temp > final_temp:
        temps.append(temp)
        eners.append(current_energy)
        bests.append(best_energy)
        idx = random.randint(0, len(current_spins) - 1)
        new_spins = current_spins.copy()
        new_spins[idx] *= -1

        new_energy = max_cut_ising_energy(new_spins, weights)

        if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / temp):
            current_spins = new_spins
            current_energy = new_energy

            if current_energy < best_energy:
                best_spins = current_spins.copy()
                best_energy = current_energy

        temp *= cooling_rate
        curr_step += 1
        if curr_step >= num_steps:
            break
    return best_spins

def max_cut_ising_solver(weights, initial_temp=1000, final_temp=1, cooling_rate=0.995, num_steps=500):
    n = len(weights)
    initial_spins = np.array([random.choice([-1, 1]) for _ in range(n)])

    optimized_spins = simulated_annealing(initial_spins, weights, initial_temp, final_temp, cooling_rate, num_steps)

    partition_s = [i for i in range(n) if optimized_spins[i] == 1]
    partition_t = [i for i in range(n) if optimized_spins[i] == -1]

    return partition_s, partition_t

# weights = np.array([[0, 1,-1, 0, -1, 6],
#                     [1,0,-0.2,-2,0,4.4],
#                     [-1,-0.2,0,4,-4.4,0],
#                     [0,-2,4,0,6,-1],
#                     [-2,0,-4.4,6,0,-0.2],
#                     [6,4.4,0,-1,-0.2,0]])
weights = np.random.randint(low=0, high=10, size=(200, 200))
partition_s, partition_t = max_cut_ising_solver(weights)
print(f"Partition S: {partition_s}")
print(f"Partition T: {partition_t}")
xs = list(range(len(bests)))
fig, ax1 = plt.subplots()
ax1.plot(xs, eners, label = "Current Energy", linewidth=0.7, color='lightsteelblue')
ax1.plot(xs, bests, label = "Best Solution's Energy", linewidth=2, color='orange')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('System Energy')
ax2 = ax1.twinx()
ax2.plot(xs, temps, label="Temperature", color='green')
ax2.set_ylabel('Temperature')
ax1.legend(loc='upper right')
ax2.legend(loc='lower left')
plt.title('Simulated Annealing for MAX-CUT Problem (Ising Model)')
```
````