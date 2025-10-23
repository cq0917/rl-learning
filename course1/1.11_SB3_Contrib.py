import gymnasium as gym

env = gym.make('CartPole-v1')

from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import TRPO
#sb3_contrib提供了一些算法,列表:https://sb3-contrib.readthedocs.io/en/master/guide/examples.html#tqc
#各个算法的适用环境:https://stable-baselines3.readthedocs.io/en/master/guide/algos.html

model = TRPO(policy='MlpPolicy', env=env, verbose=0)

# 训练
model.learn(total_timesteps=2_0000, progress_bar=True)

# 测试
from stable_baselines3.common.evaluation import evaluate_policy
result = evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)
print(result)