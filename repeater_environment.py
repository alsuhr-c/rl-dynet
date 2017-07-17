import dynet as dy
import editdistance as ed
import math
import numpy as np
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000

words = open("names.txt").read().strip().split()

class RepeaterEnvironment():
  def __init__(self):
    self.reference_string = random.choice(words)
    print("reference string: " + self.reference_string)

    self.my_string = [ ]
    self.my_prev_string = [ ]
    self.terminated = False

  def has_finished(self):
    return self.reference_string == self.my_string or self.terminated

  def take_action(self, action):
    self.my_prev_string = self.my_string
    if action == 26:
      self.terminated = True 
    elif action == 27:
      self.my_string = self.my_string[:-1]
    else:
      self.my_string.append(chr(action + 97))

  def reward(self):
    rec = 0.
    for char in self.reference_string:
      if char in self.my_string:
        rec += 1
    if len(self.reference_string) > 0:
      rec /= len(self.reference_string)

    prec = 0.
    for char in self.my_string:
      if char in self.reference_string:
        prec += 1
    if len(self.my_string) > 0:
      prec /= len(self.my_string)
    
    if prec + rec == 0.:
      f1 = 0.
    else:
      f1 = 2 * (prec * rec) / (prec + rec)

    if len(self.my_string) > len(self.reference_string):
      f1 -= 1

    return f1


  def previous_state(self):
    return self.reference_string, self.my_prev_string

  def current_state(self):
    return self.reference_string, self.my_string

class RepeaterModel():
  def __init__(self):
    self.model = dy.Model()

    # architecture: l1 embeddings for the full and current scales, and l2
    # combines the two and puts biases to transform to decision
    self.char_embs = self.model.add_lookup_parameters((26, 5))

    self.in_rnn = dy.LSTMBuilder(1, 5, 20, self.model)
    self.my_rnn = dy.LSTMBuilder(1, 5, 20, self.model)

    self.final_weights = self.model.add_parameters((40, 28)) 
    self.final_biases = self.model.add_parameters((28))

    self.empty_state = self.model.add_parameters((20))

    self.trainer = dy.AdamTrainer(self.model, alpha = 0.001)

  def forward(self, state):
    ref_str = state[0]
    cur_str = state[1]

    rnn_state = self.in_rnn.initial_state()
    for char in ref_str: 
      val = ord(char) - 97
      rnn_state = rnn_state.add_input(self.char_embs[val])

    ref_state = rnn_state.output()

    if len(cur_str) > 0:
      rnn_state = self.my_rnn.initial_state()
      for char in cur_str: 
        val = ord(char) - 97
        rnn_state = rnn_state.add_input(self.char_embs[val])

      cur_state = rnn_state.output()
    else:
      cur_state = dy.parameter(self.empty_state)

    cat = dy.concatenate([ref_state, cur_state])

    result = dy.transpose(dy.rectify(dy.reshape(cat, (1, 40)) * dy.parameter(self.final_weights) + dy.reshape(dy.parameter(self.final_biases), (1, 28))))
#    result = dy.transpose(dy.rectify(dy.reshape(cat, (1, 40)) * dy.parameter(self.final_weights)))
    
    return result

  def select_action(self, state, total_step_num):
    sample = random.random()
    threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_step_num / EPS_DECAY)

    if sample > threshold:
      dy.renew_cg()
      dist = self.forward(state).npvalue()
      return np.argmax(dist)
    else:
      return random.randint(0, 28 - 1)
