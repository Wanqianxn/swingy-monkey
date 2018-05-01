# Imports.
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
        self.gravity = 0
        self.Q = dict()
        self.lr = 0.2
        self.gamma = 1.
        self.epsilon = 0.01

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        if self.last_state is None:
            if npr.rand() < self.epsilon:
                new_action = npr.choice([0,1])
            else:
                new_action = 0
            self.last_action = new_action
            self.last_state  = state
            return self.last_action
        
#        Find gravity after the first frame
        if all([self.last_state is not None, self.gravity == 0, self.last_action == 0]):
            self.gravity = self.last_state['monkey']['vel'] - state['monkey']['vel']

    
#        blur_last_state = [tree dist, tree top - monkey top, monkey vel]
#        Estimate 3600 states for the crudest resolution. Reduce resolution when cover reaches a quarter
        screen_blur = 32
        vel_blur = 8
        blur_last_state = (self.gravity,
                           np.int(self.last_state['tree']['dist']/screen_blur),
                           np.int((self.last_state['tree']['top']-self.last_state['monkey']['top'])/screen_blur),
                           np.int(self.last_state['monkey']['vel']/vel_blur))
        blur_state = (self.gravity,
                      np.int(state['tree']['dist']/screen_blur),
                      np.int((state['tree']['top']-state['monkey']['top'])/screen_blur),
                      np.int(state['monkey']['vel']/vel_blur))


#        update Q for last state
        if blur_last_state in self.Q:
            if blur_state in self.Q:
                self.Q[blur_last_state][self.last_action] = (1-self.lr)*self.Q[blur_last_state][self.last_action] + self.lr*(self.last_reward + self.gamma*np.max(self.Q[blur_state]))
            else:
                self.Q[blur_last_state][self.last_action] = (1-self.lr)*self.Q[blur_last_state][self.last_action] + self.lr*(self.last_reward)
        else:
            if blur_state in self.Q:
                self.Q[blur_last_state] = [0,0]
                self.Q[blur_last_state][self.last_action] = (self.last_reward + self.gamma*np.max(self.Q[blur_state]))
            else:
                self.Q[blur_last_state] = [0,0]
                self.Q[blur_last_state][self.last_action] = self.last_reward
        
#        choose action
        if blur_state in self.Q:
            if npr.rand() < self.epsilon:
                new_action = npr.choice([0,1])
            else:
                new_action = np.argmax(self.Q[blur_state])
        else:
            if npr.rand() < self.epsilon:
                new_action = npr.choice([0,1])
            else:
                new_action = 0

        self.last_action = new_action
        self.last_state  = state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        
        self.last_reward = reward


def run_games(learner, hist, cover, iters = 100, t_len = 100):
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
        cover.append(len(learner.Q))

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()
    # Empty list to save history.
    hist = []
    cover = []

    # Run games. 
    run_games(agent, hist, cover, 3000, 1)

    # Save history. 
    np.save('hist',np.array(hist))
    np.save('cover',np.array(cover))
    plt.plot(hist)
    plt.show()
    plt.plot(cover)
    plt.show()


