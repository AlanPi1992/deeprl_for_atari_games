"""Main DQN agent."""

import keras
from keras.models import Model
from keras.optimizers import Adam
from deeprl_hw2.objectives import mean_huber_loss
import numpy as np
import gym
import pickle

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 window_size):
        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.window_size = window_size


    def calc_q_values(self, state, q_net):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        q_value = q_net.predict_on_batch(state)
        return q_value


    def update_policy(self, target_q):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        # mini_batch = self.preprocessor.process_batch(self.memory.sample(self.batch_size))
        mini_batch_index = self.memory.sample(self.batch_size)

        x = []
        for _sample in mini_batch_index:
            x.append(self.memory.buffer[_sample].state.astype(np.float32))
        x = np.asarray(x)
        y = self.calc_q_values(x, self.q_network) # reserve the order in mini_batch

        # print(y)

        counter = 0
        for _sample in mini_batch_index:
            if self.memory.buffer[_sample].is_terminal:
                y[counter, self.memory.buffer[_sample].action] = self.memory.buffer[_sample].reward
            else:
                _tmp = self.calc_q_values(np.asarray([self.memory.buffer[_sample].next_state.astype(np.float32),]), target_q)
                y[counter, self.memory.buffer[_sample].action] = self.memory.buffer[_sample].reward + self.gamma * max(_tmp[0])
            counter += 1

        
        # print('=================================================')
        train_loss = self.q_network.train_on_batch(x, y)
        return train_loss


    def fit(self, env, env_name, output_add, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """

        # Alaogrithm 1 from the reference paper
        # Initialize a target Q network as same as the online Q network
        config = Model.get_config(self.q_network)
        target_q = Model.from_config(config)
        loss = []
        score = []
        episode_len = []
        Q_update_counter = 0
        old_Q_update_counter = 0
        targetQ_update_counter = 0
        evaluate_counter = 0
        episode_counter = 0
        while True:
            if Q_update_counter > num_iterations:
                break
            # For every new episode, reset the environment and the preprocessor
            episode_counter += 1
            print("********  0 Begin the training episode: ", episode_counter, ", currently ", Q_update_counter, " step  *******************")
            initial_frame = env.reset()
            self.preprocessor.reset()
            prev_phi_state_n = self.preprocessor.process_state_for_network(initial_frame, initial_frame)
            prev_phi_state_m = self.preprocessor.process_state_for_memory(initial_frame, initial_frame)
            prev_frame = np.copy(initial_frame)
            for t in range(max_episode_length):
                # Generate samples according to different policy
                if self.memory.current_size > self.num_burn_in:
                    _tmp = self.calc_q_values(np.asarray([prev_phi_state_n,]), self.q_network)
                    _action = self.policy.select_action(_tmp[0], True)
                else:
                    _action = np.random.randint(0, self.policy.epsilon_greedy_policy.num_actions)
                next_frame, reward, is_terminal, debug_info = env.step(_action)
                reward = self.preprocessor.process_reward(reward)
                phi_state_n = self.preprocessor.process_state_for_network(next_frame, prev_frame)
                phi_state_m = self.preprocessor.process_state_for_memory(next_frame, prev_frame)
                self.memory.append(prev_phi_state_m, _action, reward, phi_state_m, is_terminal)
                
                # Save the trained Q-net at 4 check points
                Q_update_counter += 1
                if Q_update_counter == 1:
                    self.q_network.save(output_add + '/qnet-0of3.h5')
                elif Q_update_counter == num_iterations // 3:
                    self.q_network.save(output_add + '/qnet-1of3.h5')
                elif Q_update_counter == num_iterations // 3 * 2:
                    self.q_network.save(output_add + '/qnet-2of3.h5')
                elif Q_update_counter == num_iterations:
                    self.q_network.save(output_add + '/qnet-3of3.h5')

                # Update the Q net using minibatch from replay memory and update the target Q net
                if self.memory.current_size > self.num_burn_in:
                    # Update the Q network every self.train_freq steps
                    if Q_update_counter % self.train_freq == 0:
                        loss.append([Q_update_counter, self.update_policy(target_q)])
                        # print(self.calc_q_values(np.asarray([prev_phi_state_n,]), self.q_network)[0])
                        evaluate_counter += 1
                        # if evaluate_counter % 20000 == 0:
                        if evaluate_counter % 10000 == 0:
                            score.append([Q_update_counter, self.evaluate(env_name, 10, max_episode_length)])
                            print("1 The average total score for 10 episodes after ", evaluate_counter, " updates is ", score[-1])
                            print("2 The loss after ", evaluate_counter, " updates is: ", loss[-1])
                    # Update the target Q network every self.target_update_freq steps
                    targetQ_update_counter += 1
                    if targetQ_update_counter == self.target_update_freq:
                        targetQ_update_counter = 0
                        config = Model.get_config(self.q_network)
                        target_q = Model.from_config(config)

                prev_frame = np.copy(next_frame)
                prev_phi_state_m = np.copy(phi_state_m)
                prev_phi_state_n = np.copy(phi_state_n)
                if is_terminal:
                    break
            # Store the episode length
            episode_len.append(Q_update_counter - old_Q_update_counter)
            old_Q_update_counter = Q_update_counter
        # Save the episode_len, loss, score into files
        pickle.dump( episode_len, open( output_add + "/episode_length.p", "wb" ) )
        pickle.dump( loss, open( output_add + "/loss.p", "wb" ) )
        pickle.dump( score, open( output_add + "/score.p", "wb" ) )


# ''' ========================================================================'''
# ''' =====================  For double Q-net ================================'''
# ''' ========================================================================'''
    def update_policy_double(self, second_q):
        """Update your policy in double Q network

        """
        # mini_batch = self.preprocessor.process_batch(self.memory.sample(self.batch_size))
        mini_batch_index = self.memory.sample(self.batch_size)

        x = []
        for _sample in mini_batch_index:
            x.append(self.memory.buffer[_sample].state.astype(np.float32))
        x = np.asarray(x)

        # Randomly change the role of q network 1 and 2 and do update
        _rand = np.random.uniform()
        if _rand < 0.50:
            y = self.calc_q_values(x, self.q_network)
            tmp_action = np.argmax(y, axis = 1)

            counter = 0
            for _sample in mini_batch_index:
                if self.memory.buffer[_sample].is_terminal:
                    y[counter, self.memory.buffer[_sample].action] = self.memory.buffer[_sample].reward
                else:
                    _tmp = self.calc_q_values(np.asarray([self.memory.buffer[_sample].next_state.astype(np.float32),]), second_q)
                    y[counter, self.memory.buffer[_sample].action] = self.memory.buffer[_sample].reward + \
                                                                     self.gamma * _tmp[0][tmp_action[counter]]
                counter += 1

            train_loss = self.q_network.train_on_batch(x, y)
        else:
            y = self.calc_q_values(x, second_q)
            tmp_action = np.argmax(y, axis = 1)

            counter = 0
            for _sample in mini_batch_index:
                if self.memory.buffer[_sample].is_terminal:
                    y[counter, self.memory.buffer[_sample].action] = self.memory.buffer[_sample].reward
                else:
                    _tmp = self.calc_q_values(np.asarray([self.memory.buffer[_sample].next_state.astype(np.float32),]), self.q_network)
                    y[counter, self.memory.buffer[_sample].action] = self.memory.buffer[_sample].reward + \
                                                                     self.gamma * _tmp[0][tmp_action[counter]]
                counter += 1

            train_loss = second_q.train_on_batch(x, y)

        return train_loss


    def fit_double(self, env, env_name, output_add, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.
           Using double deep Q-network 
        """

        # Initialize a second Q network as same as the 1st Q network, and compile it
        config = Model.get_config(self.q_network)
        second_q_net = Model.from_config(config)
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        second_q_net.compile(optimizer=adam, loss=mean_huber_loss)

        # INITIALIZE counters and containers
        loss = []
        score = []
        episode_len = []
        Q_update_counter = 0
        old_Q_update_counter = 0
        targetQ_update_counter = 0
        evaluate_counter = 0
        episode_counter = 0
        while True:
            if Q_update_counter > num_iterations:
                break
            # For every new episode, reset the environment and the preprocessor
            episode_counter += 1
            print("********  0 Begin the training episode: ", episode_counter, ", currently ", Q_update_counter, " step  *******************")
            initial_frame = env.reset()
            self.preprocessor.reset()
            prev_phi_state_n = self.preprocessor.process_state_for_network(initial_frame, initial_frame)
            prev_phi_state_m = self.preprocessor.process_state_for_memory(initial_frame, initial_frame)
            prev_frame = np.copy(initial_frame)
            for t in range(max_episode_length):
                # Generate samples according to different policy
                if self.memory.current_size > self.num_burn_in:
                    _rand = np.random.uniform()
                    if _rand < 0.50:
                        _tmp = self.calc_q_values(np.asarray([prev_phi_state_n,]), self.q_network)
                    else:
                        _tmp = self.calc_q_values(np.asarray([prev_phi_state_n,]), second_q_net)
                    _action = self.policy.select_action(_tmp[0], True)
                else:
                    _action = np.random.randint(0, self.policy.epsilon_greedy_policy.num_actions)
                next_frame, reward, is_terminal, debug_info = env.step(_action)
                reward = self.preprocessor.process_reward(reward)
                phi_state_n = self.preprocessor.process_state_for_network(next_frame, prev_frame)
                phi_state_m = self.preprocessor.process_state_for_memory(next_frame, prev_frame)
                self.memory.append(prev_phi_state_m, _action, reward, phi_state_m, is_terminal)
                
                # Save the trained Q-net at 4 check points
                Q_update_counter += 1
                if Q_update_counter == 1:
                    self.q_network.save(output_add + '/qnet-0of3.h5')
                elif Q_update_counter == num_iterations // 3:
                    self.q_network.save(output_add + '/qnet-1of3.h5')
                elif Q_update_counter == num_iterations // 3 * 2:
                    self.q_network.save(output_add + '/qnet-2of3.h5')
                elif Q_update_counter == num_iterations:
                    self.q_network.save(output_add + '/qnet-3of3.h5')

                # Update the Q net using minibatch from replay memory and update the target Q net
                if self.memory.current_size > self.num_burn_in:
                    # Update the Q network every self.train_freq steps
                    if Q_update_counter % self.train_freq == 0:
                        loss.append([Q_update_counter, self.update_policy_double(second_q_net)])
                        # print(self.calc_q_values(np.asarray([prev_phi_state_n,]), self.q_network)[0])
                        evaluate_counter += 1
                        if evaluate_counter % 20000 == 0:
                        # if evaluate_counter % 100 == 0:
                            score.append([Q_update_counter, self.evaluate(env_name, 10, max_episode_length)])
                            print("1 The average total score for 10 episodes after ", evaluate_counter, " updates is ", score[-1])
                            print("2 The loss after ", evaluate_counter, " updates is: ", loss[-1])

                prev_frame = np.copy(next_frame)
                prev_phi_state_m = np.copy(phi_state_m)
                prev_phi_state_n = np.copy(phi_state_n)
                if is_terminal:
                    break
            # Store the episode length
            episode_len.append(Q_update_counter - old_Q_update_counter)
            old_Q_update_counter = Q_update_counter
        # Save the episode_len, loss, score into files
        pickle.dump( episode_len, open( output_add + "/episode_length.p", "wb" ) )
        pickle.dump( loss, open( output_add + "/loss.p", "wb" ) )
        pickle.dump( score, open( output_add + "/score.p", "wb" ) )
# ''' ========================================================================'''
# ''' =====================  End For double Q-net ============================'''
# ''' ========================================================================'''



    def evaluate(self, env_name, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """

        # Run the policy for 20 episodes and calculate the mean total reward (final score of game)
        env = gym.make(env_name)
        mean_reward = 0
        for episode in range(num_episodes):
            initial_frame = env.reset()
            self.preprocessor.reset()
            prev_phi_state_n = self.preprocessor.process_state_for_network(initial_frame, initial_frame)
            total_reward = 0
            prev_frame = np.copy(initial_frame)
            for t in range(max_episode_length):
                _tmp = self.calc_q_values(np.asarray([prev_phi_state_n,]), self.q_network)
                _action = self.policy.select_action(_tmp[0], False)
                next_frame, reward, is_terminal, debug_info = env.step(_action)
                phi_state_n = self.preprocessor.process_state_for_network(next_frame, prev_frame)
                # Use the original reward to calculate total reward
                total_reward += reward
                if is_terminal:
                    break
                prev_frame = np.copy(next_frame)
                prev_phi_state_n = np.copy(phi_state_n)
            mean_reward += total_reward
        return mean_reward/num_episodes

