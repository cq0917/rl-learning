from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation,ResizeObservation 
from my_wrapper import SkipFrameWrapper
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def make_env():    
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrameWrapper(env,skip=4)
    env= GrayScaleObservation(env,keep_dim=True)
    env = ResizeObservation(env,shape=(84,84))  
    return env

def main():
    # 训练时用什么环境,测试时就用什么环境
    # vec_env = SubprocVecEnv([make_env for _ in range(1)]) # 由于VecFrameStack()只接受并行化环境,所以采用这种形式 
    vec_env = make_vec_env(make_env, n_envs=1)  # 只评估一个环境使用这种方式即可,默认使用DummyVecEnv
    vec_env = VecFrameStack(vec_env,n_stack=4,channels_order='last')  
    model = PPO.load('best_mdoel', env=vec_env)
    reward_sum_mean,reward_sum_std = evaluate_policy(model,vec_env,n_eval_episodes=20,deterministic=True)  # n_eval_episodes表示测试多少局游戏
    print(reward_sum_mean,reward_sum_std)



# 使用sb3提供的evaluate_policy不能可视化动画,想要可视化动画需要自己手动调用env.render()
def main(model_path):
    # DummyVecEnv + VecFrameStack 让模型收到的观测形式与训练时一致
    vec_env = make_vec_env(make_env, n_envs=1)  # 只评估一个环境使用这种方式即可,默认使用DummyVecEnv
    vec_env = VecFrameStack(vec_env,n_stack=4,channels_order='last') 
    model = PPO.load("best_mdoel", env=vec_env)

    obs = vec_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        vec_env.render()
        if dones[0]:
            obs = vec_env.reset()




if __name__ =='__main__':
    main()






