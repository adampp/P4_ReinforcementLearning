import gym
from mdptoolbox import example
from hiive.mdptoolbox import mdp
from hiive.visualization import mdpviz
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

import gym_forest

import time

def evaluateRT(problem, mutate=False):
    # Enumerate state and action space sizes
    num_states = problem.observation_space.n
    num_actions = problem.action_space.n

    # Intiailize T and R matrices
    R = np.zeros((num_actions, num_states, num_states))
    T = np.zeros((num_actions, num_states, num_states))

    # Iterate over states, actions, and transitions
    for state in range(num_states):
        for action in range(num_actions):
            for transition in problem.P[state][action]:
                probability, next_state, reward, done = transition
                R[action, state, next_state] = reward
                T[action, state, next_state] = probability
            
            # Normalize T across state + action axes
            T[action, state, :] /= np.sum(T[action, state, :])

    # Conditionally mutate and return
    if mutate:
        problem.env.R = R
        problem.env.T = T
    return R, T

#############################
## forest or frozen
problem = 'forest'
## pi, vi, q
algo = 'q'
#############################
if problem == 'frozen':
    np.random.seed(1)
    mapsize = 32
    map = generate_random_map(size=mapsize, p=0.96)
    env = gym.make('FrozenLake-v0', desc=map)
    env._max_episode_steps = 1e6
elif problem == 'forest':
    env = gym.make('Forest-v0')
    env._max_episode_steps = 1e3
    
state = env.reset()
R, T = evaluateRT(env)

if algo == 'pi':
    solver = mdp.PolicyIteration(T, R, 0.9, max_iter=5000)
elif algo == 'vi':
    solver = mdp.ValueIteration(T, R, 0.9, epsilon=1e-6, max_iter=5000, initial_value=0)
elif algo == 'q':
    solver = mdp.QLearning(T, R, 0.99, alpha=1.0, alpha_decay=0.9999993, alpha_min=0.1,
        epsilon=1.0, epsilon_min=0.2, epsilon_decay=0.999999,
        n_iter=6e6, run_stat_frequency = 1e4)
        
solver.setVerbose()

start = time.time()
solver.run()
end = time.time()

if problem == 'forest':
    print(solver.policy)
elif problem == 'frozen':
    idx = 0
    for i in range(mapsize):
        row = ''
        for j in range(mapsize):
            if env.desc[i, j].decode('UTF-8') != 'F':
                row = row + env.desc[i, j].decode('UTF-8') + ' '
            elif solver.policy[i*mapsize+j] == 0:
                row = row + '< '
            elif solver.policy[i*mapsize+j] == 1:
                row = row + 'v '
            elif solver.policy[i*mapsize+j] == 2:
                row = row + '> '
            elif solver.policy[i*mapsize+j] == 3:
                row = row + '^ '
        print(row)
        
print(f"runtime={end - start} sec")

cumReward = 0.0;
cumIterations = 0.0;
for _ in range(100):
    state = env.reset()
    idx = 0
    while True:
        cumIterations += 1.0
        state, reward, is_done, test = env.step(solver.policy[state])
        cumReward += reward 
        idx += 1
        
        if is_done or idx >= env._max_episode_steps:
            break
print(f"average reward={cumReward / 100.0}")

print(f"average iterations={cumIterations / 100.0}")


