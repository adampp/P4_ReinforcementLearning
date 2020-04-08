from gym.envs.registration import register


register(
    id='Forest-v0',
    entry_point='gym_forest.envs:ForestEnv',
)