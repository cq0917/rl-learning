'''
callback类用于手动干预/监控训练过程

# Callback语法
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

        #可以访问的变量
        #self.model
        #self.training_env
        #self.n_calls
        #self.num_timesteps
        #self.locals
        #self.globals
        #self.logger
        #self.parent

    def _on_training_start(self) -> None:
        #第一个rollout开始前调用
        pass

    def _on_rollout_start(self) -> None:
        #rollout开始前
        pass

    def _on_step(self) -> bool:
        #env.step()之后调用,返回False后停止训练
        return True

    def _on_rollout_end(self) -> None:
        #更新参数前调用
        pass

    def _on_training_end(self) -> None:
        #训练结束前调用
        pass

CustomCallback()
'''


# 让训练只执行N步的callback
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

env = gym.make('CartPole-v1')

# 自定义callback
class SimpleCallback(BaseCallback):
    def __init__(self, verbose = 0):
        super().__init__(verbose)
        self.call_count = 0

    def _on_step(self):
        self.call_count += 1
        if self.call_count % 20 ==0:
            print(self.call_count)
        if self.call_count >= 100:
            return False
        return True
    
model = PPO('MlpPolicy',env,verbose=0)
model.learn(8000,callback=SimpleCallback())

'''
model.learn(..., callback=SimpleCallback()) 会先创建 SimpleCallback 的实例，并把它交给 SB3 的训练循环。
训练过程中,SB3 每执行一步环境交互(即每次 env.step() 完成并把 transition 加入 rollout)就会调用一次回调的 _on_step() 方法：
_on_step() 返回 True → 训练继续；返回 False → 立即停止 learn()。
你在 _on_step() 里把 call_count 累加，相当于记录已经执行了多少个环境 step。
每逢 call_count 是 20 的倍数就打印当前步数。
当 call_count >= 100 时返回 False,于是训练在第 100 个 step 就提前结束，即使 model.learn 要求的是 8000 个 total_timesteps。
这就是 callback 起作用的方式：训练主循环会在关键节点（开始、每步、每轮更新、结束等）调用回调对象的钩子方法，你可以在这些钩子里插入自定义逻辑，比如计数、记录指标、保存模型或主动中止训练。
'''

'''保存训练过程中性能最好的模型正是 callback 的常见用途之一。SB3 已经提供了现成的 EvalCallback / CheckpointCallback 等回调专门做这些事'''