import gym
from stable_baselines3 import A2C

env = gym.make('CartPole-v1')
model = A2C('MlpPolicy',env,verbose=1,tensorboard_log='logs')  # tensorboard已经被集成到stable-baselines3中了
model.learn(total_timesteps=1000000)
# 训练中,rollout的过程既收集了数据又相当于阶段性的评估了当前策略,这也就是ep_len_mean和ep_rew_mean两条tensorboard曲线的数据来源

obs = env.reset()  # 训练结束后环境不在初始状态处,在评估之前必须要reset

for i in range(1000):
    action,_state = model.predict(obs,deterministic=True)  # 如果未使用循环神经网络,不用管_state    deterministic默认是False
    obs,reward,done,info = env.step(action)
    env.render()
    if done:
        obs = env.reset()


'''
tensorboard训练日志:
time曲线表明训练速度  训练的快不快就看日志中的FPS
ep_len_mean和ep_rew_mean曲线是最值得关注的两条曲线  其中ep_rew_mean是未考虑discounted rate的
policy_loss表示策略网络的损失。我们期望它逐渐上升趋向于0(意味着更新幅度变小、策略稳定)
value_loss表示价值网络的损失。我们期望它逐渐下降趋向于0
entropy_loss记录的是负熵。熵越高,策略越倾向于随机探索;熵越低,策略越确定。我们期望训练过程中熵越来越小,即entropy_loss逐渐上升
explained_variance(可解释方差)是一个衡量价值网络预测准确度的指标,比value_loss更直观,期望它逐渐上升并趋近于1
KL散度通常期望在0.01-0.03之间  太大表示策略更新太剧烈,太小表示没更新
'''




# 快速训练版
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# 8 个环境并行采样
env = make_vec_env("CartPole-v1", n_envs=8)  # 多环境并行采样靠的是 CPU 性能,不是GPU

'''
如果没有显式传 vec_env_cls,它就把 vec_env_cls 设成 DummyVecEnv,无论 n_envs 是 1 还是 8。也就是说,默认情况下哪怕你要并行好几个环境，
它还是用单进程的 DummyVecEnv 把这些环境串行执行。只有你自己在调用时传入 vec_env_cls=SubprocVecEnv,才会改用多进程版本。
所以你看到的 env = make_vec_env("CartPole-v1", n_envs=8) 实际用的仍然是 DummyVecEnv。
DummyVecEnv:单进程顺序执行所有子环境，优点是调试方便(可以直接断点);缺点是无法真正并行。但对于CartPole 这类轻量环境,DummyVecEnv 往往更快
SubprocVecEnv:为每个环境起一个子进程并行运行,一般比DummyVecEnv更快
结论：环境轻、需要调试 → DummyVecEnv; 环境重 → SubprocVecEnv

n_envs 的设定主要依据你 CPU 的“核心数” (Number of CPU Cores)，但并非简单地设为核心数就最好。
我的CPU是10核16线程,我可以设定10<=n_envs<=16
可以选取几个n_envs值进行一个短时的训练,观察训练日志中的FPS(代表了每秒钟智能体与环境交互的总步数,FPS越大,训练速度越快),选择使FPS最大的n_envs
'''

model = A2C(
    policy="MlpPolicy",  # 可以自定义策略网络
    env=env,
    n_steps=64,          # 每个环境一次rollout采64步(默认是5),共512条样本  
    learning_rate=7e-4,  
    verbose=1,
    device='cpu',
    tensorboard_log="logs",
)
'''
其中有个device参数,默认是'auto',也就是有GPU时优先用GPU,没有则用CPU
经试验发现:
在环境较简单、输入(状态)非图像、模型较小(MLP)的情况下,使用CPU比GPU还快
'''

'''
n_steps越大,训练的速度越快。total_timesteps总量不变,n_steps变大,网络更新次数变少,训练时间减少。
这类似于在深度学习里把 batch_size 放大，一次迭代消耗更多样本、更新次数变少，整体训练更快
'''
model.learn(total_timesteps=200_000)  

# 评估时再单独建一个环境
import numpy as np

eval_env = gym.make("CartPole-v1")
ep_rew = []
ep_len = []
for _ in range(10):
    obs = eval_env.reset()  # 每一个episode结束之后要重新reset
    done = False
    rew,len = 0,0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        len += 1
        rew += reward
    ep_len.append(len)        
    ep_rew.append(rew)  
print(f'ep_rew_mean = {np.mean(ep_rew)} , ep_len_mean = {np.mean(ep_len)}')      



