import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

class GoLeftEnv(gym.Env):  # 自定义gym环境需要继承gym.Env环境
    metadata = {'render_modes':['console']}
    '''
    render_mode:
    human: 弹出一个实时渲染的可视化窗口,可视化智能体的行为
    rgb_array: 返回numpy 数组(代表图像的像素)。可以自己保存成图片或拼成视频;也可以用于训练视觉智能体。常在无图形界面的服务器上运行。
    大多数强化学习项目里，如果需要可视化，调试阶段会选 human;需要把过程录制下来或在无界面环境里运行时，会用 rgb_array。
    '''
    def __init__(self):
        super().__init__()
        self.pos = 9  # 初始位置
        self.action_space = gym.spaces.Discrete(2)  # 本环境只有左和右两个动作  把环境的动作空间定义成“离散的两个动作”。也就是说智能体在每一步只能选择动作集合[0, 1]之一。
        self.observation_space = gym.spaces.Box(low=0,
                                                high=10,
                                                shape=(1,),
                                                dtype=np.float32)
        
    def reset(self,seed=None,options=None):
        self.pos = 9  # 重置位置
        return np.array([self.pos],dtype=np.float32),{}
    
    def step(self,action):
        if action == 0:
            self.pos -= 1
        if action ==1:
            self.pos += 1
        
        self.pos = np.clip(self.pos,0,10)  # 位置处于[0,10]内,超出时裁剪

        done = self.pos==0  # 判断游戏结束

        reward = 1 if self.pos == 0 else 0  # 给予reward

        return np.array([self.pos],dtype=np.float32),reward,bool(done),False,{}
    
    def render(self,mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(self.pos)

    def close(self):
        pass

env = GoLeftEnv()

# 检查自定义环境是否合法
check_env(env,warn=True)  # check_env是SB3内置的函数

# 包装环境
from stable_baselines3.common.env_util import make_vec_env
train_env = make_vec_env(lambda:GoLeftEnv(),n_envs=1)
'''
train_env = make_vec_env(lambda: GoLeftEnv, n_envs=1) 是在把自定义环境包装成 SB3 需要的“向量化环境”。
PPO 等算法都要求传入 VecEn,哪怕只有 1 个环境,也要用 DummyVecEnv 裝一层;make_vec_env 正好帮你完成这件事。
'''

# 定义模型
from stable_baselines3 import PPO
model = PPO('MlpPolicy',train_env,verbose=0)

def test(model,env):
    state = env.reset()
    done = False
    step = 0

    for i in range(100):
        action,_ = model.predict(state,deterministic=True)
        next_state,reward,done,_ = env.step(action)
        print(step,state,action,reward)
        state = next_state
        step += 1
        if done:
            break

# 未训练前测试
test(model,train_env)

# 训练后再测试
# model.learn(5000)
# test(model,train_env)

