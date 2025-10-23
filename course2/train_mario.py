import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation,ResizeObservation  # gym提供了现成的灰度图及Resize包装器    环境包装器是gym的核心功能
from my_wrapper import SkipFrameWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv,VecFrameStack,DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback  # sb3自带EvalCallback及tensorboard
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''对于马里奥这个游戏,仅靠无脑增加total_timesteps解决不了这个问题,必须进行一些处理'''

def make_env():    

    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # 跳帧  用同一个action连续执行多帧  这个处理也是用于提高训练效率,而不是为了提升性能
    env = SkipFrameWrapper(env,skip=4)  

    # 由于玩马里奥这个游戏和彩色图像还是灰度图没什么区别,所以我们把彩色图像转化为灰度图像,提升训练效率    它倒不能用于提升训练效果
    env= GrayScaleObservation(env,keep_dim=True)

    # 这个过程是缩放(将临近的几个像素点合并成一个像素点)而不是裁剪  保证智能体仍然能看见完整的游戏画面,只不过变模糊了
    env = ResizeObservation(env,shape=(84,84))  
    return env

# 闭包 创建线性衰减调度器
def linear_schedule(initial_value,final_value):
    def func(progress_remaining):  # progress_remaining从1(开始)到0(结束)
        return final_value + (initial_value-final_value)*progress_remaining
    return func

def main():
    vec_env = SubprocVecEnv([make_env for _ in range(16)])  # 并行化环境
    # 将连续的几帧游戏画面堆叠在一起，形成一个单一的观测作为policy_network的输入  使策略网络的输入包含历史信息
    vec_env = VecFrameStack(vec_env,n_stack=4,channels_order='last')  # VecFrameStack是专门设计用来处理向量化环境

    # 单独创建评估环境
    eval_env = VecFrameStack(SubprocVecEnv([make_env for _ in range(4)]),n_stack=4,channels_order="last")
    eval_callback = EvalCallback(eval_env,best_model_save_path='./best_model/',log_path='./callback_logs/',eval_freq=10000//16)
    

    # 马里奥这个游戏的状态是图像,所以我们使用PPO内置的CNN模型   
    model = PPO('CnnPolicy',vec_env,verbose=1,tensorboard_log='logs',
                learning_rate=linear_schedule(3e-4,1e-5),
                n_steps=2048,  # 如果提示out of memory,优先降低n_steps/n_envs,减少缓存中存放的transition的数量
                batch_size=1024,
                n_epochs=8,
                gamma=0.95,
                gae_lambda=0.92,
                ent_coef=0.1,
                clip_range=linear_schedule(0.25,0.1),
                max_grad_norm=0.5,
                vf_coef=0.75)
    
    model.learn(total_timesteps=1e7,callback=eval_callback)

if __name__ =='__main__':
    main()




'''
Tricks:
1、训练过程可以观察tensorboard曲线,情况不对就及时停止
2、更好的做法是新创建一个环境用于评估,评估时使用确定性策略    
3、model.save('ppo_mario') 是保存最后一轮训练结束之后的权重,没用。实际中都是用callback类保存最优模型
4、SB3框架保存模型时会保存成一个zip压缩包
5、PPO超参数解析:    
    1、数据收集和使用类
    n_steps (int, default: 2048)
    含义: 在每次策略更新之前，每个并行环境要运行的步数。把收集到的所有数据(n_steps * n_envs 条)存起来，然后用这些数据来更新一次网络。 值越大,训练越快但需要更多内存
    batch_size (int, default: 64)
    含义: 在进行策略更新时,一次送入网络的小批量(mini-batch)的大小。在每次策略更新时,算法会把收集到的 n_steps * n_envs 条数据分成很多个 batch_size 大小的小批量，然后用这些小批量数据来训练网络。
    n_epochs (int, default: 10)
    含义: 收集完数据后,算法会用n_steps * n_envs 条数据完整地训练 n_epochs 遍。值越大，数据利用率越高，但有过拟合当前这批数据的风险

    2、PPO核心参数
    clip_range (float or Schedule, default: 0.2)
    含义: 用于限制新旧策略之间的变化幅度。ratio = new_policy / old_policy 会被裁剪到 [1 - clip_range, 1 + clip_range] 的范围内。防止策略大幅更新
    target_kl
    含义: 计算出的 KL 散度超过了你设定的 target_kl 阈值,那么更新循环就会被早停,即使 n_epochs 还没有跑完。举例：
    设置 n_epochs=10, target_kl=0.015。算法开始更新，跑完了第 1、2、3 个 epoch。在第 4 个 epoch 的某次 mini-batch 更新后，算法发现当前策略与更新开始前的策略之间的 KL 散度达到了 0.016。
    因为 0.016 > 0.015,“安全刹车”被触发。训练将停止对这批数据的学习，直接进入下一轮的数据收集阶段，哪怕后面还有 6 个 epoch 没有跑。
    gamma (float, default: 0.99)
    含义: 用于计算未来奖励的当前价值。gamma 越接近1,智能体越有“远见”,会更看重未来的长期回报。越接近0,智能体越“短视”,只关心眼前的奖励。
    gae_lambda (float, default: 0.95)
    含义: GAE(广义优势估计)的 lambda 参数。用于在偏差和方差之间做权衡。lambda=0 等价于TD(0)优势估计,偏差高但方差低。lambda=1 等价于蒙特卡洛优势估计,偏差低但方差高。0.95 是一个经过实践检验的、效果很好的折中值。
    normalize_advantage (bool, default: True)
    含义: 是否对优势函数进行标准化。通常能显著稳定训练过程，特别是在奖励范围变化很大的环境中。建议保持为 True。

    3、损失函数系数 (决定各部分损失的权重)
    ent_coef (float, default: 0.0)
    含义: 熵损失的系数。控制探索的强度。值越大,鼓励策略变得更随机(高熵),有利于探索。值越小,策略会更快地变得确定(低熵),有利于利用。这个值对于避免过早收敛到局部最优非常重要。
    vf_coef (float, default: 0.5)
    含义: 价值函数损失的系数。在总损失中,价值函数损失所占的比重。用于确保价值网络的训练与策略网络的训练保持同步。0.5 是一个常用的默认值。

    4、其它
    learning_rate (float or Schedule, default: 3e-4)
    含义: 学习率，决定了每次梯度下降更新的步长。 深度学习中最经典的超参数。太高会导致训练不稳定甚至发散；太低会导致训练速度过慢。可以使用一个 Schedule,让学习率随着训练的进行而衰减。
    max_grad_norm (float, default: 0.5)
    含义: 梯度裁剪(Gradient Clipping)的最大范数。将梯度的大小限制在一个阈值内，防止因梯度爆炸导致的剧烈更新，从而稳定训练。

6、optuna是常见的超参数调参工具
7、tensorboard中的曲线time/fps的值(以650为例)就表示当前每秒大约执行 650次env.step(所有并行环境总和),也就是每秒采集 650 条 (state, action, reward, next_state) transition
8、在强化学习中: "Frame"、"Timestep"(时间步)、env.step()是等价的。它指的是智能体与环境完成一次交互
9、EvalCallback 的核心作用是: 
    在训练过程中定期评估模型，并自动保存迄今为止表现最好的那个模型。
    它会在训练过程中周期性地暂停一下,然后在你指定的评估环境(这里是 eval_env)中测试当前模型的性能。
    eval_freq是评估的频率,如果想每隔n步评估一次,就设置为n//并行环境数
    EvalCallback 会比较本次评估的平均奖励和历史最佳平均奖励。如果这次更好，它就会将当前的模型保存下来，覆盖掉之前的“最佳模型”。
    并且会将评估结果记录到一个日志文件中。在 TensorBoard 中，你可以看到一条名为 eval/mean_reward 的曲线。这条曲线通常比 rollout/ep_rew_mean 平滑得多，因为它是由确定性评估产生的，能更清晰地反映模型的学习趋势。
10、包装器顺序问题: SkipFrameWrapper 套在 GrayScaleObservation 外面。跳帧时会循环调用内层 step() 多次，灰度/Resize 会在每次循环都执行一遍，白白浪费 3 次转换。
    Gym 环境包装器一层包着一层。当你调用最外层 wrapper 的 .step() 方法时，这个调用会一层一层向内传递，直到最核心的原始环境。然后，返回值再从内向外，一层一层地被处理。
11、如果有一次训练从曲线上看感觉效果不错,但是可惜当时total_timesteps设置的太小了,没关系,可以加载这次训练产生的best_model继续训练
    model = PPO.load(path=best_model.zip,env=vec_env,其余参数可任意设置)  环境不能改变
    model.learn(total_timesteps=可任意设置,callback=eval_callback)
    注意将best_model.zip另外保存,小心第二次训练把这个结果覆盖了
12、linear_schedule 是一个闭包,帮你把某些超参数(学习率、clip_range)从训练开始时的值线性衰减到结束时的值。Stable-Baselines3 允许这些参数既可以是常数，也可以是一个 callable
    在训练第 t 步时,SB3 会计算 progress_remaining = 1 - t / total_timesteps,然后调用你提供的函数得到当前值。
13、stable-baselines3 默认不显示实时进度条。要想直接看到剩余进度,可以在 learn() 时启用内置的 tqdm 进度条(需要 SB3 ≥ 2.0 且已安装 tqdm):
    model.learn(total_timesteps=int(1e7),callback=eval_callback,progress_bar=True)  # 指定progress_bar参数即可
    这样命令行里会显示一个进度条,实时展示已完成的 timesteps 和预计剩余时间。  
14、batch_size个数据就叫一个mini-batch 
15、关于训练速度的实验:
    1. n_envs越大越好,
    2. SubprocVecEnv比DummyVecEnv好
    3. 对于本任务,cpu比gpu快,看来强化学习很吃cpu
    4. batch_size和n_steps调大一定程度上可以加快训练速度,但非主要考虑因素
'''