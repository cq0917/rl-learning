import gymnasium as gym

class wrapper(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env,gym.Env),f'Expected env to be a "gymnasium.Env" type,but got {type(env)}'  # 确保一定是gym标准的环境
        super().__init__(env)
        self.step_n = 0
    
    def reset(self):
        state,info = self.env.reset()
        self.step_n = 0
        return state,info
    
    def step(self,action):
        state,reward,termination,truncation,info = self.env.step(action)

        # 一局游戏最多走N步
        self.step_n += 1
        if self.step_n >= 200:
            truncation = True

        return state,reward,termination,truncation,info

env = gym.make("CartPole-v1",)
env = wrapper(env)
env.reset()  

# torch.distributions.Categorical 创建概率分布对象
import torch
def test_dist():
    dist = torch.distributions.Categorical(torch.FloatTensor([0.1,0.2,0.7]))  # 将0/1/2三个类的概率分别设为0.1, 0.2, 0.7
    action = dist.sample()
    print('action= ',action)
    log_prob = dist.log_prob(action)
    print('log_prob= ',log_prob)

import torch
import torch.nn as nn
class PolicyNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequential = nn.Sequential(nn.Linear(4,16),
                                        nn.ReLU(),
                                        nn.Linear(16,2),
                                        nn.Softmax(dim=-1))
    
    def forward(self,state):
        state = torch.FloatTensor(state).unsqueeze(0)  # [4] -> [1,4]
        probs = self.sequential(state)
        dist = torch.distributions.Categorical(probs)  # 用softmax输出创建概率分布
        action = dist.sample()  # 在概率分布中采样获得action
        log_prob = dist.log_prob(action)  # 求动作的对数概率
        return action.item(),log_prob
policy = PolicyNetwork()

def test():
    state,_ = env.reset()  # _就是个合法的变量名,只不过约定俗成，这个变量之后不会使用  
    reward_sum = 0
    done = False

    while not done:
        action,_ = policy(state)  # 确定性策略进行评估
        state,reward,termination,truncation,_ = env.step(action)
        done = termination or truncation
        reward_sum += reward
        
    return reward_sum

def train():
    optimizer = torch.optim.Adam(policy.parameters(),lr=1e-3)
    for i in range(5000):
        rewards = []
        log_probs = []
        state,_ = env.reset()
        done = False
        # 采样一个episode的数据并存储。采一个episode数据之后就利用这些数据更新一次策略。然后用新策略再采样...
        while not done:
            action,log_prob = policy(state)  
            state,reward,termination,truncation,_ = env.step(action)
            done = termination or truncation
            # 记录这条episode中的reward和对数概率
            rewards.append(reward)
            log_probs.append(log_prob)
        
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.stack(log_probs)     
        # 对rewards进行decay之后求和
        decay = torch.arange(len(rewards))
        gamma = 0.9
        decay = gamma**decay
        rewards *= decay
        q_t = torch.tensor([sum(rewards[i:])/0.9**i for i in range(len(rewards))])
        loss = -(q_t*log_probs).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(i, sum([test() for _ in range(5)]) / 5)

train()
test()

