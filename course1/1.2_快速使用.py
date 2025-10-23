# 定义游戏环境
import gymnasium as gym
class MyWrapper(gym.Wrapper):
    def __init__(self):  # 在__init__()方法执行之前，实例就已经创建了，__init__()方法是为了给这个实例添加属性的，self指向这个实例
        env=gym.make('CartPole-v1')
        super().__init__(env)
        '''
        调用父类 (gym.Wrapper) 的 __init__ 方法，并将这个方法应用在当前正在被创建的子类实例 (MyWrapper 的实例) 上
        换句话说,self 在整个过程中指向的都是同一个对象——那个 MyWrapper 的实例。super() 只是一个“代理”或“快捷方式”，让你能够访问并执行父类中的方法,
        但操作的主体(self)始终是子类的实例
        '''

    def reset(self,seed=None,options=None):
        state,_ = self.env.reset()  
        return state, _
    
    def step(self,action):
        state,reward,termination,truncation,info= self.env.step(action)
        '''
        termination (终止): 这通常是因为智能体完成了任务(赢了)或者进入了一个失败状态(输了)。是一个真正的终点,V(s') 的值必须为 0
        truncation (截断): 意味着这一局因为一个外部限制(最常见的就是最大步数限制)而被人为地切断了,但任务本身其实还可以继续。不是真正的终点,还可以继续,V(s') 的值不能为 0,需要价值网络估算
        '''
        # done = termination or truncation
        return state,reward,termination,truncation,info
    
myenv = MyWrapper()
print(myenv.reset())  # 每次输出值不同    初始状态随机化 (Initial State Randomization)



# 使用SB3提供的强化学习模型
from stable_baselines3 import PPO
model = PPO('MlpPolicy',myenv,verbose=0)  # 参数verbose表明是否打印训练日志



## 训练前评估试一试
# from stable_baselines3.common.evaluation import evaluate_policy
# reward_sum_mean,reward_sum_std = evaluate_policy(model,env,n_eval_episodes=20)  # n_eval_episodes表示测试多少局游戏
# print(reward_sum_mean,reward_sum_std)



# SB3的训练函数
model.learn(total_timesteps=20000,progress_bar=True)  # progress_bar表示是否显示输出进度条
'''
total_timesteps表示训练步数  调用一次 env.step(action) 函数,就代表着一个 timestep 过去了
为什么使用timesteps这个概念,不指定训练多少个episode?  —— 因为episode的长度是可变的

与 Rollout 的关系:
假设 Rollout 批次大小 (n_steps) 是 2048。
model.learn(total_timesteps=20000) 开始后：
第一次 Rollout: 收集 2048 个 timesteps 的数据。
第一次 Update: 用这 2048 条数据更新策略。
第二次 Rollout: 再收集 2048 个 timesteps 的数据(总计 4096)。
第二次 Update: 更新策略。
... 这个过程会一直持续下去 ...
直到累计的 timesteps 数量达到或超过 20000,learn() 函数就会停止并返回。
'''



# SB3的评估函数
from stable_baselines3.common.evaluation import evaluate_policy
reward_sum_mean,reward_sum_std = evaluate_policy(model,myenv,n_eval_episodes=20)  # n_eval_episodes表示测试多少局游戏
print(reward_sum_mean,reward_sum_std)



