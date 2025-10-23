import gymnasium as gym

env = gym.make('CartPole-v1')

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 自定义特征抽取层
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=observation_space.shape[0],
                            out_channels=features_dim,
                            kernel_size = 1,
                            stride = 1,
                            padding = 0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features_dim,
                        out_channels=features_dim,
                        kernel_size = 1,
                        stride = 1,
                        padding = 0),
            
            torch.nn.ReLU(),
            torch.nn.Flatten(),

            torch.nn.Linear(features_dim,features_dim),
            torch.nn.ReLU())
    
    def forward(self,state):
        b = state.shape[0]
        state = state.reshape(b,-1,1,1)
        return self.sequential(state)

# 指定policy_kwargs更改默认的CnnPolicy模型的架构
model = PPO('CnnPolicy',
            env,
            policy_kwargs={
                'features_extractor_class':CustomCNN,
                'features_extractor_kwargs':{
                    'features_dim':8
                },
            },
            verbose=0)

from stable_baselines3.common.evaluation import evaluate_policy
model.learn(total_timesteps=20000,progress_bar=True)
result = evaluate_policy(model,env,n_eval_episodes=10,deterministic=False)
print(result)
