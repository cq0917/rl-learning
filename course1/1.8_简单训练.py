import gymnasium as gym

env_id = "LunarLander-v3"
env  = gym.make(env_id)

# 认识环境
def test_env():
    print('env.observation_space= ',env.observation_space)
    print('env.action_space= ',env.action_space)

    state,_ = env.reset()
    action = env.action_space.sample()
    next_state,reward,termination,truncation,_ = env.step(action)
    done = termination or truncation

    print('state= ',state)
    print('action= ',action)
    print('next_state= ',next_state)
    print('reward= ',reward)
    print('done= ',done)

test_env()  

# 定义模型    
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
model = PPO(
    policy='MlpPolicy',
    env=make_vec_env(env_id,n_envs=4),
    n_steps=1024, 
    # 每个环境在一次 roll-out 中要执行多少步  假设使用 n_envs=4（四个并行环境），
    # 那一次 roll-out 会得到 n_steps * n_envs = 1024 * 4 = 4096 条 transition。收集完这批数据之后才进入优化阶段。
    
    batch_size=64,
    # 在梯度更新时，每次从上面那 4096 条数据里取出多少条作为一个 minibatch。
    # SB3 会先把所有样本打乱，再按 batch_size 切块；例如这里 batch_size=64，就会得到 4096 / 64 = 64 个 minibatch。

    n_epochs=4,
    # 在一次 roll-out 之后，这 4096 条数据会被重复使用多少遍来更新网络。n_epochs=4 表示会把这批数据打乱 4 次，
    # 每次都按 batch_size 切块做梯度下降，也就是总共进行 4 × (4096 / 64) = 256 次参数更新。n_epochs 越大，同一批数据被利用的次数越多，但也可能带来过拟合或不稳定。
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=0
)

'''
n_epochs的解读:  和深度学习中epoch类似  同一批数据训练多少轮
先让所有并行环境跑 n_steps，得到一批大小 n_steps * n_envs 的样本；
进入优化阶段，把这批样本打乱，按 batch_size 切成若干小批次；
对同一批样本重复执行第 2 步 n_epochs 次（每次都会重新洗牌再分割）；
这一批数据用完就丢弃，重新采样下一批，重复上述步骤。
所以 n_epochs=4 就是说同一批 rollout 会被拿来训练 4 个 epoch，而不是“保留 4 批 rollout”。
'''

# 训练
model.learn(total_timesteps=200000,progress_bar=True)  # 向量化环境时，是所有环境步数的总和
'''训练循环会不断重复“收集 n_steps × n_envs 个样本 → 更新网络”，直到累计步数达到或超过 total_timesteps 为止'''
'''
向量化环境就是用同一个策略在多个环境里并行搜集数据，更新阶段会把这批样本统一打乱、按 batch_size 划分，各环境搜集到的数据混在一起来更新这个策略
也就是说，多环境只是加速采样，策略参数始终同步共享，所有环境的数据混在一起训练这一套策略。
'''

model.save('save/models/1.8')

# 测试
from stable_baselines3.common.evaluation import evaluate_policy
model = PPO.load('save/models/1.8')
result1=evaluate_policy(model,env,n_eval_episodes=10,deterministic=True)
print(result1)


# 除了自己训练模型之外,还可以加载别人分享的模型
from huggingface_sb3 import load_from_hub

model = PPO.load(
    load_from_hub('araffin/ppo-LunarLander-v2','ppo-LunarLander-v2.zip'),
    custom_objects={
        'learning_rate':0.0,
        "lr_schedule":lambda _: 0.0,
        'clip_range': lambda _: 0.0,
    },
    print_system_info=True,
)

result2=evaluate_policy(model,env,n_eval_episodes=10,deterministic=True)
print(result2)