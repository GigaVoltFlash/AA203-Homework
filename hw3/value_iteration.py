import numpy as np
import matplotlib.pyplot as plt
import pdb
import matplotlib.colors as mcolors

def w_func(x, x_eye=np.array([15, 15]), sigma=10):
    return np.exp(-np.linalg.norm(x - x_eye)**2/(2*sigma**2))
    
def value_iteration(x, V, Reward, n, m = 4, gamma=0.95):
    '''
    Action guide:
    a = 0: Up
    a = 1: Down
    a = 2: Left
    a = 3: Right
    '''
    # Setup transition matrix
    w = w_func(x)
    T = (1 - w) * np.eye(m) + w/4

    # Future states, handling edge cases here
    r = x if x[0] == n-1 else x + [1, 0]
    l = x if x[0] == 0 else x - [1, 0]
    u = x if x[1] == n-1 else x + [0, 1]
    d = x if x[1] == 0 else x - [0, 1]
    s_bar = np.vstack((u, d, l, r))

    # Value calc (203 equation uses future reward rather than current)
    state_values = V[s_bar[:, 0], s_bar[:, 1]]
    state_rewards = Reward[s_bar[:, 0], s_bar[:, 1]]

    total_value = state_rewards + gamma*state_values
    values_over_actions = T @ total_value
    value = np.max(values_over_actions)
    action = np.argmax(values_over_actions)

    return value, action
    
def next_state(x, a, m=4):
    w = w_func(x)
    T = (1 - w) * np.eye(m) + w/4

    # Future states, handling edge cases here
    r = x if x[0] == n-1 else x + [1, 0]
    l = x if x[0] == 0 else x - [1, 0]
    u = x if x[1] == n-1 else x + [0, 1]
    d = x if x[1] == 0 else x - [0, 1]
    s_bar = np.vstack((u, d, l, r))
    probs = T[a] # Get the probabilities associated with the picked state
    next_state = s_bar[np.random.choice(4, p=probs)]
    return next_state

n = 20
gamma = 0.95
x_goal = np.array([19, 9])
eps=1e-4

value_function = np.zeros((n, n))
reward_function = np.zeros((n, n))
reward_function[x_goal[0], x_goal[1]] = 1.0
optimal_actions = np.zeros((n, n), dtype=np.int32)

value_function_old = 1000 + value_function
# Iterate until convergence. When converged, the actions taken can also be taken as the optimal actions since they are
# based on the optimal value function
while np.linalg.norm(value_function.flatten() - value_function_old.flatten(), ord=np.inf) > eps:
    value_function_old = value_function.copy()
    for i in range(n):
        for j in range(n):
            value_function[i, j], optimal_actions[i, j] = value_iteration(np.array([i, j]), value_function_old, reward_function, n)

N = 100
rollouts = 1
# pdb.set_trace()
traj = np.zeros((N, rollouts, 2), dtype=np.int32)
x_init = np.array([0, 19])
traj[0] = x_init
for i in range(N-1):
    for j in range(rollouts):
        a = optimal_actions[traj[i, j, 0], traj[i, j, 1]]
        traj[i+1, j] = next_state(traj[i, j], a)


# Visualize the value function
plt.figure(figsize=(8, 6))
plt.imshow(value_function.T, cmap='viridis', origin='lower')
plt.colorbar(label='Value')
plt.title('Value Function')
plt.xlabel('x1')
plt.ylabel('x2')
x_eye = np.array([15, 15])
plt.scatter(x_goal[0], x_goal[1], color='green', label='x_goal', s=100, edgecolors='black')
plt.scatter(x_eye[0], x_eye[1], color='red', label='x_eye', s=100, edgecolors='black')
plt.legend()
plt.savefig('value_function.png')
plt.show()


# Visualize the value function
plt.figure(figsize=(8, 6))


cmap = mcolors.ListedColormap(plt.cm.viridis(np.linspace(0, 1, 4)))
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
plt.imshow(optimal_actions.T, cmap=cmap, norm=norm, origin='lower')
for j in range(rollouts):
    plt.plot(traj[:, j, 0], traj[:, j, 1], label='Trajectory'+ str(j))
cbar = plt.colorbar(ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(['Up', 'Down', 'Left', 'Right'])
plt.title('Value Function')
plt.xlabel('x1')
plt.ylabel('x2')

plt.scatter(x_init[0], x_init[1], color='blue', label='x_init', s=100, edgecolors='black')
plt.scatter(x_goal[0], x_goal[1], color='green', label='x_goal', s=100, edgecolors='black')
plt.scatter(x_eye[0], x_eye[1], color='red', label='x_eye', s=100, edgecolors='black')
plt.legend()
plt.savefig('value_iteration_rollouts.png')
plt.show()
print(traj[-1])