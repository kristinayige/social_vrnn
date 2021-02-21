import sys
import os
sys.path.append('../')
import numpy as np
import argparse
import pylab as pl
import json
import importlib

if args.gpu:
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf

#create model and feed data
model = NetworkModel(args)
batch_x, batch_vel, batch_pos,batch_goal,batch_grid, batch_ped_grid, batch_y,batch_pos_target, other_agents_pos, new_epoch = data_prep.getBatch()

input_list = []
grid_list = []
goal_list = []
ped_grid_list = []
y_ground_truth_list = []
y_pred_list = []  # uses ground truth as input at every step
other_agents_list = []
all_predictions = []
all_traj_likelihood = []
trajectories = []
batch_y = []
batch_loss = []


dict = {"batch_x": batch_x,
			        "batch_vel": batch_vel,
			        "batch_pos": batch_pos,
			        "batch_grid": batch_grid,
			        "batch_ped_grid": other_agents_info,
			        "step": step,
			        "batch_goal": batch_goal,
			        "state_noise": 0.0,
			        "grid_noise": 0.0,
			        "concat_noise": 0.0,
			        "other_agents_pos": [other_agents_pos]
			        }
feed_dict_ = model.feed_test_dic(**dict)

y_model_pred, likelihood = model.predict(sess, feed_dict_, True)

#all_predictions.append(predictions)