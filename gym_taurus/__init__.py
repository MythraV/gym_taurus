from gym.envs.registration import register

register(
    id='taurus-deb-v0',
    entry_point='gym_taurus.envs:TaurusEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','goal':'debri_res_1','headless':False,'maxval':0.1}
)
