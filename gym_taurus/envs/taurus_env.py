from gym_taurus.envs.robot import *
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
import time
import gym
from gym import spaces
import numpy as np
import os

class Taurus():
    def __init__(self):
        joint_names = {'left':[
            'Taurus_l_joint1','Taurus_l_joint2','Taurus_l_joint3','Taurus_l_joint4',
            'Taurus_l_joint5','Taurus_l_joint6','Taurus_l_joint7'
        ],
        'right': [
            'Taurus_r_joint1','Taurus_r_joint2','Taurus_r_joint3','Taurus_r_joint4',
            'Taurus_r_joint5','Taurus_r_joint6','Taurus_r_joint7'
        ]}
        self.left = Limb('Taurus_l_tip', joint_names['left'])
        self.right = Limb('r_tip', joint_names['right'])
        self.left_gripper = TwoFingerGripper(['gripper_l_joint','gripper_l_joint_m'])
        self.right_gripper = TwoFingerGripper(['gripper_r_joint','gripper_r_joint_m'])

class TaurusEnv(gym.Env):
    def __init__(self, limb='left', goal='debri_res_1', rewardfun=None, headless=False, mode='passive', maxval=0.1):
        self.pr = PyRep()
        #SCENE_FILE = '/media/mythra/Data/Mythra/Research/ForWard/gym_taurus/gym_taurus/envs/taurus_debridement_new.ttt'
        SCENE_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Taurus_debridement_new.ttt')
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        taurus = Taurus()
        if limb=='left':
            self.limb = taurus.left
        elif limb=='right':
            self.limb = taurus.right
        self.mode = mode
        if mode=='force':
            self.limb.set_joint_mode('force')
        shape_names = [goal]
        # Relevant scene objects
        self.oh_shape = [Shape(x) for x in shape_names]
        # Add tool tip
        self.oh_shape.append(self.limb.target)
        # Number of actions
        num_act = len(self.limb.joints)
        # Observation space size
        # 6 per object (xyzrpy) + 6 dimensions
        num_obs = len(self.oh_shape)*3*2
        # Setup action and observation spaces
        act = np.array( [maxval] * num_act )
        obs = np.array(          [np.inf]          * num_obs )

        self.action_space      = spaces.Box(-act,act)
        self.observation_space = spaces.Box(-obs,obs)
        if rewardfun is None:
            self.rewardfcn = self._get_reward
        else:
            self.rewardfcn = rewardfun
        self.default_config = [-1.0122909545898438, -1.5707963705062866,
                             0.9759880900382996, 0.6497860550880432,
                              1.0691887140274048, 1.1606439352035522,
                               0.3141592741012573]
        # set configs for resetting
        self.taurus_body = Shape('Taurus')
        self.taurus_body_config = self.taurus_body.get_configuration_tree()
        self.taurus_debri_config = Shape('debri_res_1').get_configuration_tree()


    def _make_observation(self):
        """Query V-rep to make observation.
           The observation is stored in self.observation
        """
        lst_o = []
        # example: include position, linear and angular velocities of all shapes
        for oh in self.oh_shape:
            lst_o.extend(oh.get_position()) 	# position
            lst_o.extend(oh.get_orientation())

        self.observation = np.array(lst_o).astype('float32');

    def _make_action(self, actions):
        """Query V-rep to make action.
           no return value
        """
        if self.mode=='force':
            # example: set a velocity for each joint
            for jnt, act in enumerate(actions):
                self.limb.set_joint_velocity(jnt, act)
        else:
            # example: set a offset for each joint
            for jnt, act in enumerate(actions):
                self.limb.offset_joint_position(jnt, act)



    def step(self, action):
        """Gym environment 'step'
        """
        # Assert the action space space contains the action
        assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
        # Actuate
        self._make_action(action)
        # Step
        self.pr.step()
        # Observe
        self._make_observation()

        # Reward
        reward = self.rewardfcn()

        # Early stop
        # if the episode should end earlier
        # done = if position outside user model space
        done = False
        if np.linalg.norm(self.observation[0:3] - self.observation[6:9])<0.02:
            done=True
        return self.observation, reward, done, {}


    def reset_tree(self):
        """Gym environment 'reset'
        """
        if self.pr.running:
            self.pr.start()

        # reset body and debri
        self.pr.set_configuration_tree(self.taurus_body_config)
        self.pr.set_configuration_tree(self.taurus_debri_config)


        self.limb.set_joint_mode(self.mode)
        # for i in range(len(self.limb.joints)):
        #     self.limb.set_joint_position(i,self.default_config[i])
        self._make_observation()
        #print(self.observation)

        # second statement may not be needed
        if not self.pr.running:
            self.pr.start()

        return self.observation

    def reset(self):
        """Gym environment 'reset' old version
        """
        if self.pr.running:
            self.pr.stop()
        self.pr.start()
        self.pr.stop()
        self.pr.start()

        self.limb.set_joint_mode(self.mode)
        # for i in range(len(self.limb.joints)):
        #     self.limb.set_joint_position(i,self.default_config[i])
        self._make_observation()
        print(self.observation)
        return self.observation

    def render(self, mode='human', close=False):
        """Gym environment 'render'
        """
        pass

    def seed(self, seed=None):
        """Gym environment 'seed'
        """
        return []

    def close(self):
        """
            Shutdown function
        """
        self.pr.stop()
        self.pr.shutdown()

    def stop(self):
        '''
            Stop simulation
        '''
        self.pr.stop()

    def start(self):
        '''
            Start simulation
        '''
        self.pr.start()

    def _get_reward(self):
        '''
            The reward function
            Put your reward function here
        '''
        reward = 1.0-2*np.linalg.norm(self.observation[0:3] - self.observation[6:9])
        return reward

if __name__ == "__main__":
    "Example usage for the TaurusRLEnv"
    env = TaurusEnv('left','debri_res_1',mode='passive', headless=False,maxval=0.1)
    print([env.limb.get_joint_position(x) for x in range(7)])
    time.sleep(1)
    for ieps in range(5):
        observation = env.reset()
        total_reward = 0
        action = env.action_space.sample()
        for t in range(10):
            #action = env.action_space.sample()
            observation, reward, done,_ = env.step(action)
            total_reward += reward
            if done:
                break
            # time.sleep(0.1)
            # print(action[0], env.limb.get_joint_position(0))
        print("Episode {} finished after {} timesteps.\tTotal reward: {}".format(ieps+1,t+1,total_reward))
    env.close()
