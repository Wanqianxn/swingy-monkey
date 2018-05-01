# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
from collections import deque
import torch
import torch.nn
import torch.autograd
import torch.optim
import random

from SwingyMonkey import SwingyMonkey

# DQN code inspired by http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class QNetwork(torch.nn.Module):

    def __init__(self, layer_sizes):
        super(QNetwork, self).__init__()
        self.layers = [torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]) 
                        for i in range(len(layer_sizes) - 1)]
        for i in range(len(self.layers)):
            self.add_module(str(i), self.layers[i])
        self.optim = torch.optim.Adam(self.parameters(), lr=0.00001)
        self.loss = torch.nn.MSELoss()

    def forward(self, phi):
        for i in range(len(self.layers) - 1):
            phi = torch.nn.functional.relu(self.layers[i](phi))
        return self.layers[-1](phi)

    def step(self, phi, target):
        self.zero_grad()
        phi = torch.autograd.Variable(torch.Tensor(phi))
        target = torch.autograd.Variable(torch.Tensor(target))
        out = self(phi)
        loss = self.loss(out, target)
        print(loss.data[0])
        loss.backward()
        self.optim.step()
        return out.data.numpy()[0]

    def eval(self, phi):
        phi = torch.autograd.Variable(torch.Tensor(phi))
        return self(phi).data.numpy()

class Learner(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.g = -1
        self.eps = 1.0
        self.qnn0 = QNetwork((14, 256, 256, 256, 256, 1))
        self.qnn1 = QNetwork((14, 256, 256, 256, 256, 1))
        self.qnn1.load_state_dict(self.qnn0.state_dict())
        self.memory = deque()
        self.memsize = 128
        self.batchsize = 8
        self.gamma = 1
        self.t = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.g = -1

    def decay_eps(self):
        self.eps = 1 / (1 + 0.01 * self.t)
        self.t += 1

    def vectorize_state(self, state, a):
        s = []
        s.append(float(state['tree']['dist']))
        s.append(float(state['tree']['top']))
        s.append(float(state['tree']['bot']))
        s.append(float(state['monkey']['vel']))
        s.append(float(state['monkey']['top']))
        s.append(float(state['monkey']['bot']))
        if a == 0:
            s.append(float(self.g))
            s.extend([0] * 7)
        else:
            s = [0] * 7 + s + [float(self.g)]
        return s

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        if self.last_action != None:
            self.memory.append((self.last_state, self.last_action, self.last_reward, state))
            if len(self.memory) > self.memsize:
                self.memory.popleft()
            batch = random.sample(self.memory, min(self.batchsize, len(self.memory)))
            phiPrevs = []
            phi0s = []
            phi1s = []
            rs = []
            for i in range(len(batch)):
                phiPrevs.append(self.vectorize_state(batch[i][0], batch[i][1]))
                phi0s.append(self.vectorize_state(batch[i][3], 0))
                phi1s.append(self.vectorize_state(batch[i][3], 1))
                rs.append(batch[i][2])
            qswings = self.qnn1.eval(phi0s)
            qjumps = self.qnn1.eval(phi1s)
            self.qnn1.load_state_dict(self.qnn0.state_dict())
            if len(qswings) == 1:
                qstars = np.array([np.max(np.hstack((qswings, qjumps)))])
            else:
                qstars = np.max(np.hstack((qswings, qjumps)), axis=1)
            rs = np.array(rs)
            targets = rs + self.gamma * qstars
            targets[rs < 0] = rs[rs < 0]
            self.qnn0.step(phiPrevs, targets)


        phi0 = self.vectorize_state(state, 0)
        phi1 = self.vectorize_state(state, 1)
        qswing = self.qnn0.eval(phi0)[0]
        qjump = self.qnn0.eval(phi1)[0]

        if self.last_action != None and self.last_action == 0:
                self.g = state['monkey']['vel'] - self.last_state['monkey']['vel']

        if npr.random() < self.eps:
            if npr.random() < 0.5:
                self.last_action = 0
            else:
                self.last_action = 1
        elif qswing > qjump:
            self.last_action = 0
        else:
            self.last_action = 1

        self.last_state = state

        self.decay_eps()

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

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
    pg.quit()
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 1000, 0)
    print(hist)

    # Save history. 
    np.save('hist',np.array(hist))














