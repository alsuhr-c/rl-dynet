import dynet as dy
import environment as env
import math
import numpy as np
import random

BLOCK_EMB_SIZE = 2

MAX_NUM_STEPS = 50
MAX_NUM_EPISODES = 20

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

GAMMA = 0.999

def affine(vec, weights, bias):
  bias = dy.parameter(bias)
  return vec * dy.parameter(weights) + dy.reshape(bias, (1, bias.dim()[0][0]))  

class Model():
  def __init__(self):
    self.model = dy.Model()

    # Embeds the five states at each square: empty, blocked, occupied by agent,
    # goal, and * (occupied by both agent and goal).
    self.emb_env_mat = self.model.add_lookup_parameters((5, BLOCK_EMB_SIZE))
    self.num_spots = env.WORLD_SIZE * env.WORLD_SIZE

    tot_size = BLOCK_EMB_SIZE * self.num_spots

    self.l1_weights = self.model.add_parameters((tot_size,
                                                 int(tot_size / 2)))
    self.l1_biases = self.model.add_parameters((int(tot_size / 2)))
    self.l2_weights = self.model.add_parameters((int(tot_size / 2),
                                                 int(tot_size / 4)))
    self.l2_biases = self.model.add_parameters((int(tot_size / 4)))
    self.l3_weights = self.model.add_parameters((int(tot_size / 4),
                                                 int(tot_size / 8)))
    self.l3_biases = self.model.add_parameters((int(tot_size / 8)))

    self.final_layer = self.model.add_parameters((int(tot_size / 8),
                                                  4))
    self.trainer = dy.AdamTrainer(self.model)

  def forward(self, environment, current_pos):
    start_pos = environment.start_pos
    goal_pos = environment.goal_pos
    ind_env = [ [ 1 if val else 0 for val in row ] for row in environment.world]
    ind_env[current_pos[0]][current_pos[1]] = 2
    ind_env[goal_pos[0]][goal_pos[1]] = 3
    if current_pos == goal_pos:
      ind_env[current_pos[0]][goal_pos[1]] = 4

    flat_env = [ ]
    for row in ind_env:
      flat_env.extend(row)

    emb_env = [ self.emb_env_mat[val] for val in flat_env ]
    emb_env = dy.reshape(dy.concatenate(emb_env),
                         (1, self.num_spots * BLOCK_EMB_SIZE)) 

    l1_val = affine(emb_env, self.l1_weights, self.l1_biases)
    l2_val = affine(l1_val, self.l2_weights, self.l2_biases)
    l3_val = affine(l2_val, self.l3_weights, self.l3_biases)

    return dy.transpose(l3_val * dy.parameter(self.final_layer))

  def select_action(self, environment, step_num, current_pos):
    possible_actions = env.possible_actions(environment.world, current_pos)

    sample = random.random()
    threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step_num / EPS_DECAY)

    if sample > threshold:
      dist = self.forward(environment, current_pos).npvalue()
      if not 0 in possible_actions:
        dist[0] = -100000
      if not 1 in possible_actions:
        dist[1] = -100000
      if not 2 in possible_actions:
        dist[2] = -100000
      if not 3 in possible_actions:
        dist[3] = -100000
      return np.argmax(dist)
    else:
      return random.choice(possible_actions)

  def optimize(self, environment, prev_pos, action, next_pos, reward):
    # Get Q(s_t, a_t): predictions of action taken in environment at
    # previous position
    q = dy.pick(self.forward(environment, prev_pos), action)

    # V: max of Q at next state
    v = dy.max_dim(self.forward(environment, next_pos))

    expval = v * GAMMA + reward

    loss = q - expval
    loss.backward()
    self.trainer.update()
    
model = Model()

episode_num = 0
total_step_num = 0
environment = env.Environment()
while episode_num < MAX_NUM_EPISODES:

  step_num = 0
  current_pos = environment.start_pos
  has_finished = False
  while not (has_finished or step_num > MAX_NUM_STEPS):
    action = model.select_action(environment, total_step_num, current_pos)
    x, y = current_pos

    prev_pos = current_pos

    if action == 0:
      current_pos = (x - 1, y)
    elif action == 1:
      current_pos = (x + 1, y)
    elif action == 2:
      current_pos = (x, y - 1)
    elif action == 3:
      current_pos = (x, y + 1)
#    elif action == 4:
#      has_finished = True

    if current_pos == environment.goal_pos:
      has_finished = True

    # Inverse Manhattan distance to goal
    distance_x = -math.fabs(current_pos[0] - environment.goal_pos[0])
    distance_y = -math.fabs(current_pos[1] - environment.goal_pos[1])

    # Maximum distance is 2 * env.WORLD_SIZE 
    max_distance = 2 * env.WORLD_SIZE

    manh_dist = float(distance_x + distance_y) / max_distance
#    reward = manh_dist

    reward = 0
    if current_pos == environment.goal_pos:
      reward = 1

    # Also give a penalty if we've terminated but haven't reached the goal
#    if current_pos != environment.goal_pos and has_finished:
#      reward -= 1.

    # Give a penalty for being slow.
    reward -= step_num * 0.02 

    model.optimize(environment, prev_pos, action, current_pos, reward) 

    total_step_num += 1
    step_num += 1
  print(str(reward) + "\t" + str(manh_dist))
