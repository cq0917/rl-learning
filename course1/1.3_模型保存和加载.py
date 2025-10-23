# 定义环境
import gymnasium as gym

class MyWrapper(gym.Wrapper):
    def __init__(self):
        env = gym.make('Pendulum-v1')
        super().__init__(env)
    
    def reset(self,seed=None,options=None):
        state,_=self.env.reset()
        return state,_
    
    def step(self,action):
        state,reward,termination,truncation,info = self.env.step(action)
        return state,reward,termination,truncation,info

env = MyWrapper()
env.reset()

# 训练并保存,再加载模型
from stable_baselines3 import PPO

# 训练
model = PPO('MlpPolicy',env,verbose=0)
model.learn(8000,progress_bar=True)

# 保存  保存模型时不会一同保存环境
model.save('save/models/model_1.3')

# 加载
model = PPO.load('save/models/model_1.3',env=env)

# 加载模型后继续训练  
model.learn(6000,progress_bar=True)

