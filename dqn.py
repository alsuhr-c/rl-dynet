import dynet as dy
import environment as env

MAX_NUM_EPISODES = 50
MAX_NUM_STEPS = 50

GAMMA = 0.999

model = env.Model()

episode_num = 0
total_step_num = 0

def optimize(model, prev_state, current_state, action, reward):
  q = dy.pick(model.forward(prev_state), action)
  v = dy.max_dim(model.forward(current_state))

  expval = v * GAMMA + reward

  loss = q - expval
  loss.backward()

  model.trainer.update()

while episode_num < MAX_NUM_EPISODES:
  environment = env.Environment()
  step_num = 0

  while not (environment.has_finished() or step_num > MAX_NUM_STEPS):
    action = model.select_action(environment, total_step_num)

    environment.take_action(action)

    reward = environment.reward()

    optimize(model,
             environment.previous_state,
             environment,
             action,
             reward)

    total_step_num += 1
    step_num += 1
