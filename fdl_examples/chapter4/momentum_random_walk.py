import numpy as np
import matplotlib.pyplot as plt

# Random walk
step_range = 10
step_choices = range(-1 * step_range, step_range + 1)
rand_walk = [np.random.choice(step_choices) for x in range(100)]

# Momentum random walk
momentum = 0.9
momentum_rand_walk = [np.random.choice(step_choices)]
for i in range(len(rand_walk) - 1):
    prev = momentum_rand_walk[-1]
    rand_choice = np.random.choice(step_choices)
    new_step = momentum * prev + (1 - momentum) * rand_choice
    momentum_rand_walk.append(new_step)

# Show results
plt.plot(rand_walk, label="No Momentum")
plt.plot(momentum_rand_walk, label="Momentum {}".format(momentum))
plt.legend()
plt.grid()
plt.show()
