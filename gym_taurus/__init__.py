from gym.envs.registration import register

register(
    id='taurus-deb-v0',
    entry_point='gym_taurus.envs:TaurusEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','goal':'debri_res_1','headless':False,'maxval':0.1}
)

register(
    id='taurus-deb-vec-v0',
    entry_point='gym_taurus.envs:SubprocVecTaurusEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','goal':'debri_res_1','headless':False,'maxval':0.1}
)

register(
    id='taurus-deb-vec-headless-v0',
    entry_point='gym_taurus.envs:SubprocVecTaurusEnv',
    reward_threshold=10.0,
    kwargs={'limb':'left','goal':'debri_res_1','headless':True,'maxval':0.1}
)