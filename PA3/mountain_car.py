'''
	1. Don't delete anything which is already there in code.
	2. you can create your helper functions to solve the task and call them.
	3. Don't change the name of already existing functions.
	4. Don't change the argument of any function.
	5. Don't import any other python modules.
	6. Find in-line function comments.

'''

import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt


class sarsaAgent():
	'''
	- constructor: graded
	- Don't change the argument of constructor.
	- You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
	- Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
	- You can add more instance variable if you feel like.
	- upper bound and lower bound are for the state (position, velocity).
	- Don't change the number of training and testing episodes.
	'''

	def __init__(self):
		self.env = gym.make('MountainCar-v0')
		self.epsilon_T1 = 0.0001
		self.epsilon_T2 = 0.000005
		self.learning_rate_T1 = 0.00008
		self.learning_rate_T2 = 0.000006
		self.numStates = 50
		self.weights_T1 = np.zeros((self.numStates, self.numStates, 3, 2), dtype=float)
		self.discount = 1.0
		self.train_num_episodes = 10000
		self.test_num_episodes = 100
		self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
		self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
		np.random.seed(100)
		self.numTilings = 5
		self.onedim = 5
		self.numTiles = (self.onedim**2)*self.numTilings
		self.n = int(self.numTiles*3)

		self.weights_T2 = np.zeros((self.n, self.n, 3, self.numTilings), dtype=float)

	'''
	- get_table_features: Graded
	- Use this function to solve the Task-1
	- It should return representation of state.
	'''

	def get_table_features(self, obs):
		delta0 = (self.upper_bounds[0] - self.lower_bounds[0])/self.numStates
		delta1 = (self.upper_bounds[1] - self.lower_bounds[1])/self.numStates
		position = (obs[0] - self.lower_bounds[0])/delta0
		velocity = (obs[1] - self.lower_bounds[1])/delta1
		if position >= self.numStates:
			position = self.numStates - 1
		elif position < 0:
			position = 0
		if velocity >= self.numStates:
			velocity = self.numStates - 1
		elif velocity < 0:
			velocity = 0
		return [position, velocity]

	'''
	- get_better_features: Graded
	- Use this function to solve the Task-2
	- It should return representation of state.
	'''

	def get_better_features(self, obs):
		positionmovementvalue = -0.225/self.numTilings
		velocitymovementvalue = -0.0175/self.numTilings
		position = obs[0] + 1.2
		velocity = obs[1] + 0.07
		tileindices = [-1]*self.numTilings

		for i in range(self.numTilings):
			positionMovementConstant = i * positionmovementvalue
			velocityMovementConstant = i * velocitymovementvalue
			
			xcoord = int(self.onedim * (position- positionMovementConstant)/1.7)
			ycoord = int(self.onedim * (velocity- velocityMovementConstant)/0.14)

			tileindices[i] = i*(self.onedim**2) + (ycoord*self.onedim + xcoord)
		return np.array(tileindices)
	
	
	
	'''
	- choose_action: Graded.
	- Implement this function in such a way that it will be common for both task-1 and task-2.
	- This function should return a valid action.
	- state representation, weights, epsilon are set according to the task. you need not worry about that.
	'''

	def choose_action(self, state, weights, epsilon):
		if np.random.random() < epsilon:
			return self.env.action_space.sample()
		else:
			action = np.argmax(weights[int(state[0]), int(state[1])].dot(state))
			return action

	'''
	- sarsa_update: Graded.
	- Implement this function in such a way that it will be common for both task-1 and task-2.
	- This function will return the updated weights.
	- use sarsa(0) update as taught in class.
	- state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
	'''

	def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):
		weights[int(state[0]), int(state[1]), int(action)] += learning_rate*(reward + weights[int(new_state[0]), int(new_state[1]), int(new_action)].dot(np.array(new_state)) - weights[int(state[0]), int(state[1]), int(action)].dot(np.array(state)))*(np.array(state))
		return weights

	'''
	- train: Ungraded.
	- Don't change anything in this function.

	'''

	def train(self, task='T1'): #scenes
		if (task == 'T1'):
			get_features = self.get_table_features
			weights = self.weights_T1
			epsilon = self.epsilon_T1
			learning_rate = self.learning_rate_T1
		else:
			get_features = self.get_better_features
			weights = self.weights_T2
			epsilon = self.epsilon_T2
			learning_rate = self.learning_rate_T2
		reward_list = []
		plt.clf()
		plt.cla()
		for e in range(self.train_num_episodes):
			print(e)
			current_state = get_features(self.env.reset())
			done = False
			t = 0
			new_action = self.choose_action(current_state, weights, epsilon)
			while not done:
				action = new_action
				obs, reward, done, _ = self.env.step(action)
				new_state = get_features(obs)
				new_action = self.choose_action(new_state, weights, epsilon)
				weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
											weights)
				current_state = new_state
				if done:
					reward_list.append(-t)
					break
				t += 1
		self.save_data(task)
		reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
		plt.plot(reward_list)
		plt.savefig(task + '.jpg')

	'''
	   - load_data: Ungraded.
	   - Don't change anything in this function.
	'''

	def load_data(self, task):
		return np.load(task + '.npy')

	'''
	   - save_data: Ungraded.
	   - Don't change anything in this function.
	'''

	def save_data(self, task):
		if (task == 'T1'):
			with open(task + '.npy', 'wb') as f:
				np.save(f, self.weights_T1)
			f.close()
		else:
			with open(task + '.npy', 'wb') as f:
				np.save(f, self.weights_T2)
			f.close()

	'''
	- test: Ungraded.
	- Don't change anything in this function.
	'''

	def test(self, task='T1'):
		if (task == 'T1'):
			get_features = self.get_table_features
		else:
			get_features = self.get_better_features
		weights = self.load_data(task)
		reward_list = []
		for e in range(self.test_num_episodes):
			current_state = get_features(self.env.reset())
			done = False
			t = 0
			while not done:
				action = self.choose_action(current_state, weights, 0)
				obs, reward, done, _ = self.env.step(action)
				new_state = get_features(obs)
				current_state = new_state
				if done:
					reward_list.append(-1.0 * t)
					break
				t += 1
		return float(np.mean(reward_list))

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--task", required=True,
	   help="first operand", choices={"T1", "T2"})
	ap.add_argument("--train", required=True,
	   help="second operand", choices={"0", "1"})
	args = vars(ap.parse_args())
	task=args['task']
	train=int(args['train'])
	agent = sarsaAgent()
	agent.env.seed(0)
	np.random.seed(0)
	agent.env.action_space.seed(0)
	if(train):
		agent.train(task)
	else:
		print(agent.test(task))
