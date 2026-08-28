# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```
```
Python 3.x
Gymnasium
NumPy
Matplotlib
Google Colab / Jupyter Notebook

```
## Environment Description
```
env_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

```



## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm
### Procedure

1. Initialize the **FrozenLake-v1** environment using Gymnasium.

2. Create the **Q-table** and initialize all state-action values with zeros.

3. Set the **learning parameters** such as episodes, learning rate (α), discount factor (γ), and exploration rate (ε).

4. Start each episode and select actions using the **ε-greedy policy**.

5. Record the **states, actions, and rewards** obtained during the episode.

6. Calculate the **return \(G_t\)** for each state-action pair based on the collected rewards.

7. Update the Q-table using:
   ```
          Q(s,a)←Q(s,a)+α[Gt−Q(s,a)]
   ```
8. Gradually decrease **ε** after each episode and repeat the training process.

9. Generate the final **policy** by selecting the action with the highest Q-value and calculate:
   ```
   V(s)=maxaQ(s,a)
   ```

10. Display the **Q-table, state-value function, learned policy, average reward, and learning curve** to evaluate the agent.



## Python Program
```
-------------------------------------------------
#### Monte Carlo Control


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 4100
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    if np.random.random() < epsilon:
        # Exploration
        return env.action_space.sample()
    else:
        # Exploitation
        return np.argmax(Q[state])


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate complete episode
    episode = generate_episode(epsilon)

    # Store total reward
    total_reward = sum(
        reward for state, action, reward in episode
    )

    episode_rewards.append(total_reward)

    # Calculate returns
    G = 0
    visited = set()

    for state, action, reward in reversed(episode):

        G = gamma * G + reward

        # Update Q-value
        if (state, action) not in visited:

            visited.add((state, action))

            Q[state, action] += alpha * (
                G - Q[state, action]
            )

    # Reduce epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)

state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("Learned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(optimal_policy)


# -------------------------------------------------
# Average Reward
# -------------------------------------------------

success_rate = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    success_rate
)


# -------------------------------------------------
# Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")

plt.grid(True)
plt.show()

env.close()



```


## Output


<img width="458" height="505" alt="image" src="https://github.com/user-attachments/assets/119d8e6f-c851-4902-a2d4-bafebe22e217" />

<img width="746" height="401" alt="image" src="https://github.com/user-attachments/assets/32fc1764-9bfc-4340-9514-849af2fd8e29" />



## Result

Increasing the number of training episodes from 2,100 to 4,000 improved the performance of the Monte Carlo Control agent. The agent obtained better and more stable rewards after additional training, showing improved learning and decision-making ability.





## Inference


The results indicate that **more training episodes provide greater experience** to the agent. Training for **4,000 episodes** helped the agent learn more accurate Q-values and develop a better policy compared with **2,100 episodes**, resulting in more consistent performance.




