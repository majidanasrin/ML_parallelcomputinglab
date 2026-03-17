# ======================================================
# Import Required Libraries
# ======================================================

import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt


# ======================================================
# Question A: Introduction to OpenAI Gym
# Operation: Define a custom maze environment
# ======================================================

class MazeEnv(gym.Env):

    def __init__(self):

        super(MazeEnv, self).__init__()

        # Maze layout
        # 0 = empty path
        # 1 = wall
        # 2 = goal

        self.maze = np.array([
            [0,0,0,1,0],
            [1,1,0,1,0],
            [0,0,0,0,0],
            [0,1,1,1,0],
            [0,0,0,2,0]
        ])

        self.start_pos = (0,0)
        self.agent_pos = self.start_pos

        self.goal_pos = (4,3)

        # Action space
        # 0 = up
        # 1 = down
        # 2 = left
        # 3 = right
        self.action_space = spaces.Discrete(4)

        # Observation space
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)


# ======================================================
# Question B: Environment Functions
# Operation: Reset and render the environment
# ======================================================

    def reset(self):

        self.agent_pos = self.start_pos
        return np.array(self.agent_pos)


    def render(self):

        maze_copy = self.maze.copy()

        x,y = self.agent_pos
        maze_copy[x,y] = 9

        print(maze_copy)


# ======================================================
# Question C: Defining Actions and Observations
# Operation: Move the agent inside the maze
# ======================================================

    def step(self, action):

        x,y = self.agent_pos

        if action == 0:  # up
            x -= 1
        elif action == 1: # down
            x += 1
        elif action == 2: # left
            y -= 1
        elif action == 3: # right
            y += 1

        # Boundary check
        if x < 0 or x >= 5 or y < 0 or y >= 5:
            x,y = self.agent_pos

        # Wall check
        if self.maze[x,y] == 1:
            x,y = self.agent_pos

        self.agent_pos = (x,y)

        reward = -1
        done = False

        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True

        return np.array(self.agent_pos), reward, done, {}


# ======================================================
# Question D: Implementing Q-Learning
# Operation: Initialize Q-table and parameters
# ======================================================

env = MazeEnv()

state_size = 25
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))

learning_rate = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 500

rewards_list = []


# ======================================================
# Operation: Convert state position to index
# ======================================================

def state_to_index(state):
    return state[0]*5 + state[1]


# ======================================================
# Question E: Training the Agent
# Operation: Train using Q-learning algorithm
# ======================================================

for episode in range(episodes):

    state = env.reset()
    state_index = state_to_index(state)

    total_reward = 0

    done = False

    while not done:

        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_index])

        next_state, reward, done, _ = env.step(action)

        next_state_index = state_to_index(next_state)

        q_table[state_index, action] = q_table[state_index, action] + learning_rate * (
            reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action]
        )

        state_index = next_state_index
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards_list.append(total_reward)


print("Training Finished")


# ======================================================
# Operation: Plot learning progress
# ======================================================

plt.plot(rewards_list)
plt.title("Training Rewards")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()


# ======================================================
# Question F: Evaluation and Testing
# Operation: Test trained agent
# ======================================================

state = env.reset()

done = False

print("\nAgent Path:")

while not done:

    env.render()

    state_index = state_to_index(state)

    action = np.argmax(q_table[state_index])

    state, reward, done, _ = env.step(action)

env.render()

print("Goal Reached!")