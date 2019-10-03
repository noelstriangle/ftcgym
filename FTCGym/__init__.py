from gym.envs.registration import register

register(
    id='ftcgym-v0',
    entry_point='FTCGym.envs:FTCGymEnv',
)