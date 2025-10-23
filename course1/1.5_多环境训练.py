import gymnasium as gym

env = gym.make('CartPole-v1')

'''
SB3中有两种进行多环境训练的wrapper:
DummyVecEnv: 在单线程中运行多个环境
SubprocVecEnv: 在多线程中运行多个环境
'''

from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3 import PPO
from  stable_baselines3.common.evaluation import evaluate_policy
import time

# 闭包
def make_env(env_id: str, seed: int = 0):
    """返回一个可供 VecEnv 调用的环境工厂函数。"""
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _init


def test_multiple_env(dumm,N):
    env_fns = [make_env("Pendulum-v1", seed=i) for i in range(N)]
    if dumm:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)
    
    start = time.time()

    model = PPO('MlpPolicy',env,verbose=0)
    model.learn(total_timesteps=5000)
    print('消耗时间 = ',time.time() - start)

    result = evaluate_policy(model,env,n_eval_episodes=10)
    env.close()
    return result

if __name__ == "__main__":
    result = test_multiple_env(False, 5)
    print(result)

