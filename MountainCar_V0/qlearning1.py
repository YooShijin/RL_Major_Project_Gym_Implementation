import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")

# Q-Learning settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 3000  # Render every 3000 episodes

# Discretization settings
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Epsilon-Greedy Exploration
epsilon = 1.0  # Start with high exploration
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

# Initialize Q-table with small random values
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))  # Convert to integer tuple

for episode in range(EPISODES):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    render = episode % SHOW_EVERY == 0  # Only render every SHOW_EVERY episodes

    while not done:
        # Choose action: Exploit or Explore
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])  # Best action from Q-table
        else:
            action = np.random.randint(0, env.action_space.n)  # Random action

        # Take action and observe the result
        new_state, reward, done, _, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        # Update Q-table
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 10  # Large reward for reaching the goal

        discrete_state = new_discrete_state

    # Epsilon decay (multiplicative decay)
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon *= 0.995  # Slowly reduce exploration

env.close()
