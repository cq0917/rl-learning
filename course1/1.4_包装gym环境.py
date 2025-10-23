import gymnasium as gym

# 限制最大步数wrapper
class StepLimitWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.env.reset()
    
    def step(self,action):
        over = False
        self.current_step += 1
        state,reward,termination,truncation,info = self.env.step(action)
        done = termination or truncation
        if self.current_step >= 100:
            over = True
        return state,reward,done,info,over

# 定义一个环境测试函数
def test1(env,wrap_action_in_list=False):
    print(env)
    state,info = env.reset()
    while True:
        action = env.action_space.sample()  # 返回值是一个numpy数组 
        next_state,reward,done,info,over = env.step(action)
        if env.current_step % 20 ==0:
            print(env.current_step,state,action,reward)

        if over:
            break
        state = next_state

env=gym.make('CartPole-v1')
test1(StepLimitWrapper(env))



# 修改动作空间
import numpy as np
class NormalizeActionWrapper(gym.Wrapper):
    def __init__(self, env):
        action_space = env.action_space  # env.action_space获取动作空间
        assert isinstance(action_space, gym.spaces.Box), "动作空间不连续"  # 动作空间必须连续的    gym.spaces.Box代表连续空间,gym.spaces.Discrete代表离散空间
        '''
        断言语法:
        assert <条件>, <错误提示信息>(可选)
        <条件> : 这是一个会返回 True 或 False 的表达式。
        <错误提示信息> : 这是一个字符串。如果断言失败，这个字符串会作为错误信息显示出来，帮助你更快地定位问题。
        如果条件为 True: 程序会静默地继续向下执行,就像这行代码不存在一样
        如果条件为 False: 程序会立刻停止运行,并抛出一个名为 AssertionError 的异常。如果你提供了 <错误提示信息>，它会一并显示出来。
        assert与if-raise的作用很相似,但是使用场景不同
        '''

        # 重新定义动作空间,在正负一之间的连续值
        env.action_space = gym.spaces.Box(low=-1,high=1,shape=action_space.shape,dtype=np.float32)
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = action * 2.0
        return self.env.step(action)

def test2(env):
    state,info = env.reset()
    step = 0
    while True:
        action = env.action_space.sample()  # 返回值是一个numpy数组 
        next_state,reward,termination,truncation,info = env.step(action)
        done = termination or truncation
        if step % 20 == 0:
            print(step,state,action,reward)
        if done:
            break
        state = next_state
        step += 1

env = gym.make('Pendulum-v1')
test2(NormalizeActionWrapper(env))



# 修改状态空间
import numpy as np
class StateStepWrapper(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env.observation_space, gym.spaces.Box), "状态空间不连续"

        # 原状态空间是三维向量,现在增加一维变为四维  env.observation_space获取状态空间    
        low = np.concatenate([env.observation_space.low, [0.0]])
        high = np.concatenate([env.observation_space.high, [1.0]])

        env.observation_space = gym.spaces.Box(low=low,
                                               high=high,
                                               dtype=np.float32)
        super().__init__(env)
        self.step_current = 0

    def reset(self):
        self.step_current = 0
        return np.concatenate([self.env.reset()[0], [0.0]])  # 新增加的一维在初始状态时为0

    def step(self, action):
        self.step_current += 1
        state, reward, termination, truncation, info = self.env.step(action)
        done = termination or truncation
        if self.step_current >= 100:
            done = True

        return self.get_state(state), reward, done, info

    def get_state(self, state):
        #添加一个新的state字段
        state_step = self.step_current / 100

        return np.concatenate([state, [state_step]])

def test3(env):
    state = env.reset()
    step = 0
    while True:
        action = env.action_space.sample()  # 返回值是一个numpy数组 
        next_state,reward,done,info = env.step(action)
        if step % 20 == 0:
            print(step,state,action,reward)
        if done:
            break
        state = next_state
        step += 1 

env = gym.make('Pendulum-v1')
test3(StateStepWrapper(env))



#  SB3内置的wrapper: Monitor    在训练过程中增加日志
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make("Pendulum-v1")
env = DummyVecEnv([lambda: Monitor(env)])
A2C('MlpPolicy',env,verbose=1).learn(1000)



# SB3内置的wrapper: VecNormalize  对state和reward进行Normalize
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv  # DummyVecEnv是SB3 的“向量化环境”实现 

env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
env = VecNormalize(env)