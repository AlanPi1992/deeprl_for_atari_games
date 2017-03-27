''' Using Moniter to evaluate model '''

import argparse
import os
import random
import time

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Dropout,
                          Permute)
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras import losses
import gym
from gym import wrappers
from PIL import Image

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import PreprocessorSequence


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-evaluate')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-evaluate{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Evaluate model using Monitor')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='deepQ', help='Directory to save data to')
    parser.add_argument('--seed', default=703, type=int, help='Random seed')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    print(args.output)
    os.makedirs(args.output, exist_ok=True)

    
    env = gym.make(args.env)
    env = wrappers.Monitor(env, args.output)

    # Initialize a preporcessor sequence object
    atari_preprocessor = tfrl.preprocessors.AtariPreprocessor((84, 84))
    history_preprocessor = tfrl.preprocessors.HistoryPreprocessor(4)
    preprocessor = tfrl.preprocessors.PreprocessorSequence(atari_preprocessor, history_preprocessor)

    # Initialize a policy
    _policy = tfrl.policy.GreedyEpsilonPolicy(0.05, env.action_space.n)
    policy = tfrl.policy.LinearDecayGreedyEpsilonPolicy(_policy, 1, 0.1, 1000000)

    print('load trained model...')
    # q_net = load_model('Enduro-v0-train1-1of3.h5', custom_objects={'mean_huber_loss': mean_huber_loss})
    q_net = load_model('deepQ/SpaceInvaders-v0-run2/qnet-0of3.h5', custom_objects={'mean_huber_loss': mean_huber_loss})


    mean_reward = 0
    num_episodes = 20
    for episode in range(num_episodes):
        initial_frame = env.reset()
        preprocessor.reset()
        prev_phi_state_n = preprocessor.process_state_for_network(initial_frame, initial_frame)
        total_reward = 0
        prev_frame = np.copy(initial_frame)
        for t in range(100000):
        	env.render()
        	_tmp = q_net.predict_on_batch( np.asarray([prev_phi_state_n,]) )
        	_action = policy.select_action(_tmp[0], False)
        	next_frame, reward, is_terminal, debug_info = env.step(_action)
        	# print(_tmp[0])
        	# print(_action, reward, is_terminal, debug_info)
        	phi_state_n = preprocessor.process_state_for_network(next_frame, prev_frame)
        	total_reward += reward
        	print(total_reward)
        	if is_terminal:
        		print("Episode finished after {} timesteps".format(t+1))
        		break
        	prev_frame = np.copy(next_frame)
        	prev_phi_state_n = np.copy(phi_state_n)
        mean_reward += total_reward

    print(mean_reward/num_episodes)



if __name__ == '__main__':
	main()