import dynet as dy
import gym
import math
import numpy as np
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

class CartEnvironment():
  def __init__(self):
    self.env = gym.make('CartPole-v0')
    self.obs = self.env.reset()
    self.prev_obs = None
    self.done = False
    self.r = 0
    self.num_steps = 0

  def has_finished(self):
    return self.done

  def take_action(self, action):
    self.num_steps += 1
    self.prev_obs = self.obs
    self.obs, self.r, self.done, info = self.env.step(action)

  def reward(self):
    return self.r #int(not self.done)

  def previous_state(self):
    return self.prev_obs

  def current_state(self):
    return self.obs

class CartModel():
  def __init__(self):
    self.model = dy.Model()

    self.w_1 = self.model.add_parameters((4, 4))
    self.b_1 = self.model.add_parameters((4))
    self.w_2 = self.model.add_parameters((4, 2))

    self.trainer = dy.AdamTrainer(self.model, alpha = 0.001)

  def forward(self, state):
    # State should be a length-four matrix
    l1 =dy.reshape(dy.inputTensor(state),
                               (1,
                                4)) * dy.parameter(self.w_1) + dy.reshape(dy.parameter(self.b_1),
                                                                          (1, 4))
    l2 = l1 * dy.parameter(self.w_2)
    
    return dy.transpose(l2)

  def select_action(self, state, total_step_num):
    sample = random.random()
    threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_step_num / EPS_DECAY)

    if sample > threshold:
      dy.renew_cg()
      dist = self.forward(state).npvalue()
      return np.argmax(dist)
    else:
      return random.randint(0, 1)
