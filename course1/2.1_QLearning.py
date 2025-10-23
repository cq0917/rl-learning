'''
我们常常会采用继承 gym.wrapper 类将环境定义成包装器的形式(这样定义出的环境同样是标准的gym环境),而不是继承 gym.env 直接定义环境。除非这个环境需要我们从零开始手动定义。
环境可以用多个包装器嵌套包装
'''
import gymnasium as gym

class wrapper(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env,gym.Env),f'Expected env to be a "gymnasium.Env" type,but got {type(env)}'
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

env = gym.make("FrozenLake-v1",
               render_mode='rgb_array',
               map_name= '4x4',
               is_slippery = False)
env = wrapper(env)
env.reset()  # 实例化之后,第一步永远都是调用env.render()

'''
—— 我可以把一个gym标准的环境包装成一个不符合gym标准的环境吗? 比如我可以重写reset方法,gym标准里reset方法返回state和info,重写之后,只返回state,这样简洁一些。实际中经常会这样做吗?
def reset(self):
state,_ = self.env.reset()
return state
——
技术上，完全可以这么做。但实践中，这是一种强烈不推荐的做法，并且几乎没有人会这样做，因为它会带来严重的问题。
从纯粹的 Python 编程角度来看,当然可以继承一个类(gym.Wrapper),然后重写<改变其行为和返回值。Python 语言本身不会阻止你。我重写的 reset 方法本身是合规的。
如果你后续不调用stable-baselines3等其他依赖库,就没问题。但是一旦使用了其他按照gym标准设计的库,就会报错:
所有主流的强化学习框架，如 Stable Baselines3 (SB3), Tianshou, CleanRL 等, 它们都是针对gym标准的环境设计的
它们的内部代码都严格假定 env.reset() 会返回一个包含两个元素的元组 (state, info)

保险起见: 以后定义库时,一定要符合gym标准
'''

## 可视化环境
# from matplotlib import pyplot as plt
# def show():
#     plt.figure(figsize=(8,8))  
#     plt.imshow(env.render())  
#     # 当render_mode被设置为rgb_array时,调用env.render()不会弹出任何窗口,它会返回一个numpy数组(代表图像)
#     # imshow()函数用于将numpy数组显示未图像
#     plt.show()
# show()

## 认识环境
# def test_env():
#     print('env.observation_space= ',env.observation_space)
#     print('env.action_space= ',env.action_space)

#     state,_ = env.reset()
#     action = env.action_space.sample()
#     next_state,reward,termination,truncation,_ = env.step(action)
#     done = termination or truncation

#     print('state= ',state)
#     print('action= ',action)
#     print('next_state= ',next_state)
#     print('reward= ',reward)
#     print('done= ',done)
# test_env()

# 定义q_table  大小为状态数*动作数  存放action_value
import numpy as np
q_table = np.zeros([16,4])

# 获得动作  epsilon-greedy
import random

def get_action(state,eps):
    if random.random() < eps:
        return env.action_space.sample()
    return q_table[state].argmax()

# 测试
def test():
    state,_ = env.reset()  # _就是个合法的变量名,只不过约定俗成，这个变量之后不会使用  
    reward_sum = 0
    done = False

    while not done:
        action = get_action(state,0)  # 确定性策略进行评估
        state,reward,termination,truncation,_ = env.step(action)
        done = termination or truncation
        reward_sum += reward
        
    return reward_sum

# 训练  更新q_table
gamma = 0.95
lr = 0.7
def train():
    eps = np.linspace(1,0.01,10000)  # eps逐渐衰减
    for i in range(10000):
        state,_ = env.reset()
        done = False
        while not done:
            action = get_action(state,eps[i])
            next_state,reward,termination,truncation,_ = env.step(action)
            done = termination or truncation
            if not termination:
                TD_error = q_table[state][action] - [reward+gamma*q_table[next_state].max()]
            else:
                TD_error = q_table[state][action] - reward
            q_table[state][action] -= lr*TD_error
            state = next_state
        
        if i % 50 == 0:
            print(i,np.mean([test() for _ in range(5)]))
    np.savetxt('save/2.1_QLearning',q_table)  # 保存q_table即保存了策略

train()

q_table = np.loadtxt('save/2.1_QLearning')














