import gym
class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    
    # 跳帧是要重写step()这个方法
    def step(self,action):
        total_reward = 0
        for _ in range(self._skip):
            obs,reward,done,info = self.env.step(action)
            total_reward += reward
            if done:  # 判断中间是否done
                break
        return obs,total_reward,done,info  # 返回得到的奖励之和