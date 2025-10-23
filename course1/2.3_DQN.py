import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np

env = gym.make('CartPole-v0')
env.reset()

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4,128),
    nn.ReLU(),
    nn.Linear(128,2)
)

# 引入目标网络,解决漂移问题    延迟更新
next_model = nn.Sequential(
        nn.Linear(4,128),
        nn.ReLU(),
        nn.Linear(128,2)
)

# 把model的参数赋值给next_model
next_model.load_state_dict(model.state_dict())

import random

def get_action(state):
    if random.random() < 0.1:
        return random.choice([0,1])  # 等概率地从0与1之间随机抽样一个action
    state = torch.FloatTensor(state).reshape(-1,4)
    return model(state).argmax().item()

# 引入经验回放池
datas = []
def update_data():  # 向经验池中添加N条新数据,删除M条最古老的数据
    old_count = len(datas)
    # 新增数据
    while len(datas) - old_count < 200:
        state,_ = env.reset()
        over = False
        while not over:
            action = get_action(state)
            next_state,reward,termination,truncation,_ = env.step(action)
            over = termination or truncation
            datas.append((state,action,reward,next_state,over))
            state = next_state

    update_count = len(datas) - old_count
    drop_count = max(len(datas)-10000,0)

    while len(datas) > 10000:  # 经验回放池上限是10000,超出时从最古老的开始删除
        datas.pop(0)

    return update_count,drop_count

def get_sample():
    samples = random.sample(datas,64)  # 从经验回放池中随机采样64个样本
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1,4)
    action = torch.LongTensor([i[1] for i in samples]).reshape(-1,1)
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1,1)
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1,4)
    over = torch.LongTensor([i[4] for i in samples]).reshape(-1,1)
    return state,action,reward,next_state,over

def get_action_value(state,action):
    state = torch.FloatTensor(state).reshape(-1,4)
    output = model(state)
    action_value = output.gather(dim=1,index=action)  # gather()方法根据索引提取出我们想要的数据
    return action_value

# 计算TD_Target  需要用next_model 计算
def TD_Target(reward,next_state,over):
    next_state = torch.FloatTensor(next_state).reshape(-1,4)
    with torch.no_grad():
        action_values = next_model(next_state)

    max_action_value = action_values.max(dim=1)[0].reshape(-1,1)
    gamma = 0.98
    TD_Target =reward + gamma*max_action_value*(1-over)
    return TD_Target

def test():
    state,_ = env.reset()  # _就是个合法的变量名,只不过约定俗成，这个变量之后不会使用  
    reward_sum = 0
    done = False
    while not done:
        state = torch.FloatTensor(state).reshape(-1,4)
        action= model(state).argmax().item()  # 确定性策略进行评估
        state,reward,termination,truncation,_ = env.step(action)
        done = termination or truncation
        reward_sum += reward        
    return reward_sum

def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-3)
    loss = nn.MSELoss()

    for epoch in range(500):
        update_count,drop_count = update_data()  # 更新经验回放池数据
        # 更新过数据之后,开始学习
        for i in range(200):
            state,action,reward,next_state,over = get_sample()  # 采样一批数据用于学习
            model_output = get_action_value(state,action)
            ground_truth = TD_Target(reward,next_state,over) 
            l = loss(model_output,ground_truth)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                next_model.load_state_dict(model.state_dict())
        if epoch % 50 == 0:
            test_result = np.mean([test() for _ in range(20)])
            print(epoch,test_result)

train()
test()


















































