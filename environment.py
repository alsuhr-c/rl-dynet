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
  left = current_x > 0 and world[current_x - 1][current_y]
  right = current_x < WORLD_SIZE - 1 and world[current_x + 1][current_y]
  up = current_y > 0 and world[current_x][current_y - 1]
  down = current_y < WORLD_SIZE - 1 and world[current_x][current_y + 1]

  return left, right, up, down 

def find_path(world, current, end):
  left, right, down, up = possible_actions(world, current)

  current_x = current[0]
  current_y = current[1]

  found = current == end

  new_world = copy.deepcopy(world)

  if left:
    print("at " + str(current_x) + "," + str(current_y) + " going left")
    new_pos = (current_x - 1, current_y)
    new_world[new_pos[0]][new_pos[1]] = False
    found = found or find_path(new_world, new_pos, end)
  elif right:
    print("at " + str(current_x) + "," + str(current_y) + " going right")
    new_world = copy.deepcopy(new_world)
    new_pos = (current_x + 1, current_y)
    new_world[new_pos[0]][new_pos[1]] = False
    found = found or find_path(new_world, new_pos, end)
  elif up:
    print("at " + str(current_x) + "," + str(current_y) + " going up")
    new_world = copy.deepcopy(new_world)
    new_pos = (current_x, current_y - 1)
    new_world[new_pos[0]][new_pos[1]] = False
    found = found or find_path(new_world, new_pos, end)
  elif down:
    print("at " + str(current_x) + "," + str(current_y) + " going down")
    new_world = copy.deepcopy(new_world)
    new_pos = (current_x, current_y + 1)
    new_world[new_pos[0]][new_pos[1]] = False
    found = found or find_path(new_world, new_pos, end)

  print_world(new_world, current, end)

  return found


class Environment():
  def __init__(self):
    has_path = False
    while not has_path:
      print("Generating world")
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

      print_world(self.world, self.start_pos, self.goal_pos)

      has_path = find_path(copy.deepcopy(self.world),
                           self.start_pos,
                           self.goal_pos)
   
env = Environment()
env.print()
