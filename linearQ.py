import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt
from SwingyMonkey import SwingyMonkey


class Learner(object):

	def __init__(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.is_new = True

		# Hyperparameters and Q-values.
		self.Qvalues = dict()
		self.eta = 0.005 # Learning rate
		self.epsilon = 1 # Exploration/exploitation ratio
		self.gamma = 1 # Discount rate


	def reset(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.is_new = True


	def computeQ(self, s, a):
		''' Compute approximate Q-value from weights and features. '''
		feat = self.dk(s)
		nfeat = len(feat)
		if (feat, a) not in self.Qvalues:
			self.Qvalues[(feat, a)] = [0 for i  in range(nfeat)]
		w = self.Qvalues[(feat, a)]
		return sum([w[n] * feat[n] for n in range(nfeat)])


	def action_policy(self, state):
		''' Select action using epsilon-greedy policy.'''
		if npr.rand() < self.epsilon:
			action = self.computeQ(state, 0) < self.computeQ(state, 1)
		else:
			action = npr.randint(2)
		return int(action)

	def dk(self, state):
		''' Represent state as tuple of discretized features. '''
		f1 = int(state['tree']['dist'] / 50)
		f2 = int(state['tree']['top'] / 50)
		f3 = int(state['tree']['bot'] / 50)
		f4 = int(state['monkey']['vel'] / 5)
		f5 = int(state['monkey']['top'] / 50)
		f6 = int(state['monkey']['bot'] / 50)
		return tuple([f1, f2, f3, f4, f5, f6, 1, self.gravity])

	def action_callback(self, state):
		''' Select action (1 to jump, 0 not to) based on policy and update Q-values. '''

		# Reset gravity for each new epoch.
		if self.is_new:
			self.gravity = 0

		# Execute epsilon-greedy policy based on current state.
		new_action = self.action_policy(state)

		# Update Q-values from previous iteration.
		if self.last_state != None and self.last_action != None:
			self.is_new = False
			if self.gravity == 0:
				diff = abs(state['monkey']['vel'])
				self.gravity = 1 + 3 * int(diff > 2)
			oldQ = self.computeQ(self.last_state, self.last_action)
			correction = (self.last_reward + self.gamma * max(self.computeQ(state, 0), self.computeQ(state, 1))) - oldQ
			oldFeat = self.dk(self.last_state)
			for i in range(len(self.Qvalues[(oldFeat, self.last_action)])):
				self.Qvalues[(oldFeat, self.last_action)][i] += self.eta * correction * oldFeat[i]

		# Define policy for current iteration
		self.last_action = new_action
		self.last_state  = state

		return self.last_action

	def reward_callback(self, reward):
		'''This gets called so you can see what reward you get. '''
		self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''
	for ii in range(iters):
		# Make a new monkey object.
		swing = SwingyMonkey(sound=False,                  # Don't play sounds.
							 text="Epoch %d" % (ii),       # Display the epoch on screen.
							 tick_length = t_len,          # Make game ticks super fast.
							 action_callback=learner.action_callback,
							 reward_callback=learner.reward_callback)

		# Loop until you hit something.
		while swing.game_loop():
			pass

		# Save score history.
		hist.append(swing.score)

		# Reset the state of the learner.
		learner.reset()

		if ii == 999:
			print("after 1000:", max(hist), np.mean(hist))

	pg.quit()
	return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	N = 1000
	sg = run_games(agent, hist, N, 1)

	# Save history.
	np.savetxt('scores.csv',np.array(hist), fmt='%d')
	print("Highest Score Reached: {}".format(max(hist)))
	plt.scatter(range(N), hist)
	plt.xlabel('Iteration')
	plt.ylabel('Score')
	plt.savefig('plot.png')



