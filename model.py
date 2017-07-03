import dynet as dy
import environment as env
import math
import random

MAX_NUM_STEPS = 250
MAX_NUM_EPISODES = 20

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

def affine(vec, weights, bias):
  return vec * weights + bias 

class Model():
  def __init__(self):
    self.model = dy.Model()

    # Embeds the five states at each square: empty, blocked, occupied by agent,
    # goal, and * (occupied by both agent and goal).
    self.emb_env_mat = self.model.add_lookup_parameters((5, 2))
    self.num_spots = env.WORLD_SIZE * env.WORLD_SIZE

    tot_size = 5 * self.num_spots

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
                                                  5))
    self.trainer = dy.AdamTrainer(self.model)

  def forward(self, environment):
    start_pos = environment.start_pos
    goal_pos = environment.goal_pos
    ind_env = [ [ 1 if val else 0 for val in row ] for row in environment.world]
    ind_env[start_pos[0]][start_pos[1]] = 2
    ind_env[goal_pos[0]][goal_pos[1]] = 3
    if start_pos == goal_pos:
      ind_env[start_pos[0]][goal_pos[1]] = 4

    flat_env = [ ]
    for row in ind_env:
      flat_env.extend(row)

    emb_env = [ self.emb_env_mat[val] for val in flat_env ]
    emb_env = dy.reshape(emb_env, (1, self.num_spots)) 

    l1_val = affine(emb_env, self.l1_weights, self.l1_biases)
    l2_val = affine(l1_val, self.l2_weights, self.l2_biases)
    l3_val = affine(l2_val, self.l3_weights, self.l3_biases)

    return l2_val * self.final_layer

  def select_action(self, environment, step_num, current_pos):
    possible_actions = env.possible_actions(environment.world, current_pos)

    sample = random.random()
    threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * step_num / EPS_DECAY)

    if sample > threshold:
      dist = self.forward(environment).npvalue()
      if not 0 in possible_actions:
        dist[0] = 0
      if not 1 in possible_actions:
        dist[1] = 0
      if not 2 in possible_actions:
        dist[2] = 0
      if not 3 in possible_actions:
        dist[3] = 0

      return np.argmax(dist)
    else:
      return random.choice(possible_actions)
    
model = Model()

episode_num = 0
total_step_num = 0
while episode_num < MAX_NUM_EPISODES:
  environment = env.Environment()

  step_num = 0
  current_pos = environment.start_pos
  has_finished = False
  while not (has_finished or step_num > MAX_NUM_STEPS):
    env.print_world(environment.world, current_pos, environment.goal_pos)
    action = model.select_action(environment, total_step_num, current_pos)
    x, y = current_pos

    if action == 0:
      print("Going left")
      current_pos = (x - 1, y)
    elif action == 1:
      print("Going right")
      current_pos = (x + 1, y)
    elif action == 2:
      print("Going up")
      current_pos = (x, y - 1)
    elif action == 3:
      print("Going down")
      current_pos = (x, y + 1)
    elif action == 4:
      print("Finishing")
      has_finished = True

    total_step_num += 1
    step_num += 1
