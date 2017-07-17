import dynet as dy
import editdistance as ed
import math
import numpy as np
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

words = open("names.txt").read().strip().split()

class RepeaterEnvironment():
  def __init__(self):
    self.reference_string = random.choice(words)
    print("reference string: " + self.reference_string)

    self.my_string = [ ]
    self.my_prev_string = [ ]

  def has_finished(self):
    return self.reference_string == self.my_string

  def take_action(self, action):
    self.my_prev_string = self.my_string
    self.my_string.append(chr(action + 97))

  def reward(self):
    # string edit distance over maximum possible edit distance
    max_ed = ed.eval(self.reference_string, [])
    this_ed = ed.eval(self.reference_string, self.my_string)

    return max_ed - this_ed / max_ed

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

    self.final_weights = self.model.add_parameters((40, 26)) 
    self.final_biases = self.model.add_parameters((26))

    self.trainer = dy.AdamTrainer(self.model, alpha = 0.001)

  def forward(self, state):
    ref_str = state[0]
    cur_str = state[1]

    rnn_state = self.in_rnn.initial_state()
    for char in ref_str: 
      val = ord(char) - 97
      rnn_state.add_input(self.char_embs[val])

    ref_state = rnn_state.output()

    rnn_state = self.in_rnn.initial_state()
    for char in cur_str: 
      val = ord(char) - 97
      rnn_state.add_input(self.char_embs[val])

    cur_state = rnn_state.output()

    cat = dy.concatenate([ref_state, cur_state])

    result = dy.transpose(dy.rectify(dy.reshape(cat, (1, 40)) * dy.parameter(self.final_weights) + self.final_biases))
    
    return result

  def select_action(self, state, total_step_num):
    sample = random.random()
    threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_step_num / EPS_DECAY)

    if sample > threshold:
      dy.renew_cg()
      dist = self.forward(state).npvalue()
      return np.argmax(dist)
    else:
      return random.randint(0, 26 - 1)
