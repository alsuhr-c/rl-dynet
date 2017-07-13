import dynet as dy
import math
import numpy as np
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

MAX_NUM_ITEMS = 5

possible_items = { "PENNY": 1,
                   "NICKEL": 5,
                   "DIME": 10,
                   "QUARTER": 25 }

actions_ids = list(possible_items.keys())

### In a ScalesEnvironment, the agent should choose objects to place on the
### scale at each step. The agent is given a set of scales where one side is
### filled with items already, and the agent can place items on the other side
### of the scale, which is initially empty. 
class ScalesEnvironment():
  def __init__(self):
    self.full_side = [ ]
    self.full_weight = 0
    num_items = random.randint(1, MAX_NUM_ITEMS)
    for item in range(num_items):
      item_type = random.choice(actions_ids)
      self.full_side.append(item_type)
      self.full_weight += possible_items[item_type]

    self.empty_side = [ ]
    self.empty_weight = 0

    self.prev_empty_side = [ ]
    self.prev_empty_weight = 0

  def has_finished(self):
    return self.full_weight <= self.empty_weight 

  def take_action(self, action):
    self.prev_empty_side = self.empty_side
    self.prev_empty_weight = self.empty_weight

    action_name = actions_ids[action]
    self.empty_side.append(action_name)
    self.empty_weight += possible_items[action_name]

  def reward(self):
    if self.empty_weight <= self.full_weight:
      return float(self.empty_weight) / self.full_weight
    elif self.empty_weight <= 2 * self.full_weight:
      return float(self.empty_weight - self.full_weight) / self.full_weight
    else:
      return 0.

  def previous_state(self):
    return self.full_side, self.prev_empty_side

  def current_state(self):
    return self.full_side, self.empty_side

class ScalesModel():
  def __init__(self):
    self.model = dy.Model()

    # architecture: l1 embeddings for the full and current scales, and l2
    # combines the two and puts biases to transform to decision
    self.empty_state = self.model.add_parameters((len(possible_items)))

    self.l1_weights = self.model.add_lookup_parameters((len(possible_items),
                                                        len(possible_items)))
    self.l2_weights = self.model.add_parameters((len(possible_items) * 2,
                                                 len(possible_items)))

    self.trainer = dy.AdamTrainer(self.model)

  def forward(self, state):
    full_side = state[0]
    empty_side = state[1]

    full_embs = [self.l1_weights[actions_ids.index(item)] for item in full_side]
    empty_embs = [self.l1_weights[actions_ids.index(item)] for item in empty_side]

    full_sum = dy.esum(full_embs)

    if len(empty_embs) > 0:
      empty_sum = dy.esum(empty_embs)
    else:
      empty_sum = dy.parameter(self.empty_state)

    cat = dy.concatenate([full_sum, empty_sum])

    result = dy.transpose(dy.reshape(cat, (1, len(possible_items) * 2)) * dy.parameter(self.l2_weights))
    
    return result

  def select_action(self, state, total_step_num):
    sample = random.random()
    threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_step_num / EPS_DECAY)

    if sample > threshold:
      dist = self.forward(state).npvalue()
      return np.argmax(dist)
    else:
      return random.randint(0, len(possible_items) - 1)
