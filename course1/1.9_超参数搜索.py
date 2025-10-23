import gymnasium as gym

env_id = 'CartPole-v1'

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# 创建训练环境及测试环境  一个并行的 env_train 用于高效训练，一个带监控的 env_test 用于评估。
env_train=make_vec_env(env_id,n_envs=4)
env_test = Monitor(gym.make(env_id))
print(env_train,env_test)

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 测试一组超参数
def test_params(params):
    model = PPO(
        policy='MlpPolicy',
        env=env_train,
        n_steps=1024,
        batch_size=64,
        n_epochs=params['n_epochs'],
        gamma=params['gamma'],
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=0
    )

    model.learn(total_timesteps=params['total_timesteps'],progress_bar=True)

    mean_reward,std_reward = evaluate_policy(model,env_test,n_eval_episodes=50,deterministic=True)
    score = mean_reward - std_reward
    return score

test_params({"n_epochs":2,"gamma":0.99,"total_timesteps":500})



import optuna  # optuna用于超参数搜索
from optuna.samplers import TPESampler

# 定义一个超参数学习器
study = optuna.create_study(sampler=TPESampler(),  # TPESampler是一种比随机搜索更智能的算法。它会根据已经尝试过的参数和结果，有倾向性地选择下一组更有可能成功的参数进行尝试。
                            study_name='mark1',  # study_name仅仅是标签,便于识别
                            direction='maximize')

# 定义给 optuna 的目标函数
def f(trial):
    params = {
        'n_epochs':trial.suggest_int('n_epochs',3,5),
        'gamma':trial.suggest_uniform('gamma',0.99,0.9999),
        'total_timesteps':trial.suggest_int('total_timesteps',500,2000)
    }

    return test_params(params)
'''
optuna 在每次“尝试”（trial）时都会调用这个函数。trial 对象是 optuna 传给我们的，用于定义本次尝试的超参数。
trial.suggest_int('n_epochs', 3, 5): 告诉 optuna，对于名为 'n_epochs' 的参数，请从 [3, 4, 5] 这个整数范围中建议一个值。
trial.suggest_uniform('gamma', 0.99, 0.9999): 对于 'gamma' 参数，请在 0.99 到 0.9999 之间建议一个均匀分布的浮点数值。
params 字典被动态地创建，包含了 optuna 这次建议的一组超参数。
return test_params(params): 调用我们之前写好的核心函数，用 optuna 建议的这组参数去训练和评估，并把最终的 score 返回给 optuna。
'''

study.optimize(f,n_trials=5)
'''
study.optimize(f, n_trials=5): 启动优化过程。
它会调用函数 f 共 n_trials=5 次。
在每次调用中，optuna 的采样器会生成一组新的超参数，然后 f 函数会训练、评估并返回分数。
optuna 会记录下每一次尝试的参数和对应的分数。
'''

# 优化结束后，study 对象中保存了所有结果
print(study.best_trial.values,study.best_trial.params)

# 用最优超参数训练模型  
test_params(study.best_trial.params)