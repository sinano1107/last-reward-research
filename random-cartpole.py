# ランダムにカートポール問題に取り組む。

import random # type: ignore
import gymnasium

env = gymnasium.make('CartPole-v1', render_mode='human')

total_reward = 0
state, _ = env.reset()
done = False
truncated = False

while not done and not truncated:
    action = random.randrange(2)
    state, reward, done, truncated, _ = env.step(action)
    total_reward += reward

print('total_reward: {}'.format(total_reward))