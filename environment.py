import copy
import random

WORLD_SIZE = 10
OBS_PROB = 0.15

def print_world(world, current_pos, goal_pos):
 print(" " + "".join([str(i) for i in range(WORLD_SIZE)]))
 for y in range(WORLD_SIZE):
   row = str(y)
   for x in range(WORLD_SIZE):
     if world[x][y]:
       if goal_pos == (x, y):
         if current_pos == goal_pos:
           row += "*"
         else:
           row += "G"
       elif current_pos == (x, y):
         row += "S"
       else:
         row += " "
     else:
       row += "X"
   print(row)

def possible_actions(world, current_pos):
  current_x, current_y = current_pos
  actions = [ ] 

  # Left
  if current_x > 1 and world[current_x - 1][current_y]: 
    actions.append(0)
  # Right
  if current_x < WORLD_SIZE - 2 and world[current_x + 1][current_y]:
    actions.append(1)
  # Up
  if current_y > 1 and world[current_x][current_y - 1]:
    actions.append(2)
  # Down
  if current_y < WORLD_SIZE - 2 and world[current_x][current_y + 1]:
    actions.append(3)

  # Terminate
  actions.append(4)

  return actions 

class Environment():
  def __init__(self):
    self.world = [ [ False for _ in range(WORLD_SIZE) ] for __ in range(WORLD_SIZE)]
    empty_pos = [ ]

    for x in range(WORLD_SIZE):
      for y in range(WORLD_SIZE):
        no_obs = random.random() > OBS_PROB
        self.world[x][y] = no_obs

        if no_obs:
          empty_pos.append((x, y))

    self.goal_pos = random.choice(empty_pos)
    self.start_pos = random.choice(empty_pos)
