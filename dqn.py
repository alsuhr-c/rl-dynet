import dynet as dy
import cart_environment as env
import math

MAX_NUM_EPISODES = 100000
MAX_NUM_STEPS = 500

GAMMA = 0.999

model = env.CartModel()

episode_num = 0
total_step_num = 0
batch_size = 8

def compute_loss(model, prev_state, current_state, action, reward, step_num):
  q = dy.pick(model.forward(prev_state), action)
  v = dy.max_dim(model.forward(current_state))

  expval = v * math.pow(GAMMA, step_num) + reward

  loss = q - expval
  return loss

def batch_optimize(model, batch):
  dy.renew_cg()
  loss = [ ]
  for item in batch:
    loss.append(compute_loss(model, item[0], item[1], item[2], item[3], item[4]))

  loss = dy.esum(loss)
  loss.forward()
  loss.backward()

  model.trainer.update()

avg_reward = 0
batch = [ ]
while episode_num < MAX_NUM_EPISODES:
  step_num = 0
  environment = env.CartEnvironment()

  while not (environment.has_finished() or step_num > MAX_NUM_STEPS):
    action = model.select_action(environment.current_state(), total_step_num)

    environment.take_action(action)

    reward = environment.reward()

    batch.append((environment.previous_state(), environment.current_state(), action, reward, total_step_num))

    if len(batch) == batch_size:
      batch_optimize(model, batch)
      batch = [ ]

    total_step_num += 1
    step_num += 1

#  batch_optimize(model, batch)

  episode_num += 1

  avg_reward += step_num

  if episode_num % 100 == 0:
    print(avg_reward / 100)
    avg_reward = 0
