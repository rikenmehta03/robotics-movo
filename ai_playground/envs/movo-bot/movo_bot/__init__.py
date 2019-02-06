from gym.envs.registration import register
 
register(
    id='movobot-v0',
    entry_point='movo_bot.envs:MovobotEnv',
)