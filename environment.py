import random

MAX_NUM_ITEMS = 5

possible_items = { "PENNY": 1,
                   "NICKEL": 5,
                   "DIME": 10,
                   "QUARTER": 25 }

### In a ScalesEnvironment, the agent should choose objects to place on the
### scale at each step. The agent is given a set of scales where one side is
### filled with items already, and the agent can place items on the other side
### of the scale, which is initially empty. 
class ScalesEnvironment():
  def __init__(self):
    full_side = [ ]
    full_weight = 0
    num_items = randomm.randint(0, MAX_NUM_ITEMS)
    for item in range(num_items):
      item_type = random.choice(possible_items.keys())
      full_side.append(item_type)
      full_weight += possible_items[item_type]

    empty_side = [ ]
