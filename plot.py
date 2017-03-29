""" Plot the preformance """

import numpy as np
import pickle
import matplotlib.pyplot as plt


if __name__ == '__main__':
	# directory = 'Final_Results/SpaceInvaders-v0-run4-DeepQ/'
	# directory = 'Final_Results/SpaceInvaders-v0-run2-DoubleQ/'
	directory = 'Final_Results/SpaceInvaders-v0-run2-DuelQ/'
	# directory = 'Final_Results/SpaceInvaders-v0-run2-LinearDoubleQ/'
	# directory = 'Final_Results/SpaceInvaders-v0-run1-LinearQ/'

	# directory = 'deepQ/Enduro-v0-run37/'
	
	checkpoint_num = '4'
	name_index = 1
	network_name = ['Deep Q-network', 'Double Deep Q-network', 'Duel Deep Q-network', 
					'Linear Double Q-network', 'Linear Q-network', 'Linear Q-network (no replay and target)']

	total_score = np.asarray(pickle.load( open(directory + 'score-' + checkpoint_num + 'of5.p', 'rb') ))
	loss = np.asarray(pickle.load( open(directory+'loss-' + checkpoint_num + 'of5.p', 'rb') ))
	episode_len = pickle.load( open(directory+'episode_length-' + checkpoint_num + 'of5.p', 'rb') )
	fig = plt.figure()
	fig.suptitle(network_name[name_index])
	plot_score = fig.add_subplot(3, 1, 1)
	plot_score.plot(total_score[:,0], total_score[:,1], 'o-')
	# plot_score.set_ylim([0, 500])
	plot_score.set_ylabel('Average total reward')
	plot_score.set_xlabel('Trainging steps/interactions')
	plot_score = fig.add_subplot(3, 1, 2)
	plot_score.plot(loss[:,0], loss[:,1], 'o-')
	# plot_score.set_xlim([0, 500])
	plot_score.set_ylabel('Training loss')
	plot_score.set_xlabel('Trainging steps/interactions')
	plot_score = fig.add_subplot(3, 1, 3)
	plot_score.plot(range(0, len(episode_len), 1), episode_len[0::1], '.-')
	# plot_score.set_xlim([0, 500])
	plot_score.set_ylabel('Episode length')
	plot_score.set_xlabel('Episode number')
	plt.show()
