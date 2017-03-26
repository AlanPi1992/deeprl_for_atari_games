#!/usr/bin/env python
"""Run Atari Environment with double-DQN."""
import argparse
import os
import random
import time

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input, Dropout,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam
from keras import losses
import gym
from PIL import Image

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import PreprocessorSequence

def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    # Using tensorflow name scope
    # Create a deep Q-network
    with tf.name_scope(model_name):
        input_img = Input(shape = (window,) + input_shape) # Input shape = (4, 84, 84)
        conv1 = Convolution2D(32, (8,8), strides=4, padding='same', activation='relu')(input_img)
        # conv1 = Dropout(0.2)(conv1)
        conv2 = Convolution2D(64, (4,4), strides=2, padding='same', activation='relu')(conv1)
        # conv2 = Dropout(0.2)(conv2)
        # conv2 = Convolution2D(64, (3,3), strides=1, padding='same', activation='relu')(conv2)
        flat = Flatten()(conv2) # Flatten the convoluted hidden layers before full-connected layers
        full = Dense(512, activation='relu')(flat)
        out = Dense(num_actions)(full) # output layer has node number = num_actions
        model = Model(input = input_img, output = out)
    return model


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
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='double-deepQ', help='Directory to save data to')
    parser.add_argument('--seed', default=703, type=int, help='Random seed')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    args.output = '/home/thupxd/deeprl_for_atari_games/' + args.output # Comment out when running locally!
    os.makedirs(args.output, exist_ok=True)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    # Make the environment
    env = gym.make(args.env)
    # input('**************************  Hit to begin training...  ******************************')

    # Create a Q network
    num_actions = env.action_space.n
    q_net = create_model(4, (84, 84), num_actions, model_name='Double_Deep_Q_Net')
    # print('======================== Keras Q-network model is created. =========================')

    # Initialize a preporcessor sequence object
    atari_preprocessor = tfrl.preprocessors.AtariPreprocessor((84, 84))
    history_preprocessor = tfrl.preprocessors.HistoryPreprocessor(4)
    preprocessor_seq = tfrl.preprocessors.PreprocessorSequence(atari_preprocessor, history_preprocessor)
    # print('======================== Preprocessor object is created. =========================')

    # Initialize a replay memory
    replay_memory = tfrl.core.ReplayMemory(1000000, 4)
    # print('======================== Replay_memory object is created. =========================')

    # Initialize a policy
    _policy = tfrl.policy.GreedyEpsilonPolicy(0.05, num_actions)
    policy = tfrl.policy.LinearDecayGreedyEpsilonPolicy(_policy, 1, 0.1, 1000000)
    # print('======================== (linear-decay) Eps-Greedy Policy object is created. =========================')

    # Initialize a DQNAgent
    DQNAgent = tfrl.dqn.DQNAgent(q_net, preprocessor_seq, replay_memory, policy, gamma=0.99,
                                 target_update_freq=10000, num_burn_in=75000, train_freq=4, 
                                 batch_size=32, window_size=4)
    # print('======================== DQN agent is created. =========================')

    # Compiling, Training, Test
    # print('======================== Model compilation begin! =========================')
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    q_net.compile(optimizer=adam, loss=mean_huber_loss)
    # print('======================== Model compilation finished! =========================')
    # print('======================== Model training begin! =========================')
    DQNAgent.fit_double(env, args.env, args.output, 5000000, 100000)
    # print('======================== Model training finished! =========================')


if __name__ == '__main__':
    main()
