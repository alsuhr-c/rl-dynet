import dynet as dy
import environment as env

MAX_NUM_EPISODES = 10000
MAX_NUM_STEPS = 50

GAMMA = 0.999

model = env.ScalesModel()

episode_num = 0
total_step_num = 0

def optimize(model, prev_state, current_state, action, reward):
  q = dy.pick(model.forward(prev_state), action)
  v = dy.max_dim(model.forward(current_state))

  expval = v * GAMMA + reward

  loss = q - expval
  loss.backward()

  model.trainer.update()

avg_reward = 0
while episode_num < MAX_NUM_EPISODES:
  step_num = 0
  environment = env.ScalesEnvironment()

  reward = 1.

  while not (environment.has_finished() or step_num > MAX_NUM_STEPS):
    action = model.select_action(environment.current_state(), total_step_num)

    environment.take_action(action)

    reward = environment.reward()

    optimize(model,
             environment.previous_state(),
             environment.current_state(),
             action,
             reward)

    total_step_num += 1
    step_num += 1
  episode_num += 1

  avg_reward += reward

  if episode_num % 100 == 0:
    print(avg_reward / 100)
    avg_reward = 0
