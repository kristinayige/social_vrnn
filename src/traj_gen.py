import importlib
import json
import pylab as pl
import argparse
import numpy as np
import sys
import os
sys.path.append('../')


if args.gpu:
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf

args.dataset = '/' + args.scenario + '.pkl'
data_prep = dhlstm.DataHandlerLSTM(args)

# past velocity: batch_vel
# other agents' location: other_agents_pos
# batch_ped_grid: other_agents_info - pedestrian_grid
# local occupancy grid: self.input_grid_placeholder/ batch_grid

# predictions: tbp, batch_size, batch_size, 144
# compute_trajectory_prediction_mse: output trajectory

# in agent container
# load_map - extract the local occupancy grid


def fillVel(vel_vec):
	batch_vel = []
	for i in range(0, self.prev_horizon):
		current_vel = np.array([vel_vec[i, 0], trajectory.vel_vec[i, 1]])
		batch_vel[i*2:(i+1)*2] = np.array([current_vel[0], current_vel[1]])
	return batch_vel

# other agent positions:
# n_agents * 2
# other agent velocities:
# n_anegts * 2
def other_agent_vel_pos(other_velocities, other_positions, current_vel, current_pos):
	agent_num = other_velocities.shape[0]
	other_poses_ordered = np.zeros((other_positions.shape[0], 6))
	other_poses_ordered[:, :2] = other_positions
	other_poses_ordered[:, 2:4] = other_velocities
	for ag_id in range(agent_num):
		other_poses_ordered[ag_id, 4] = np.linalg.norm(other_poses_ordered[ag_id, :2] - current_pos)
		# ordered ids
		other_poses_ordered[ag_id, 5] = ag_id
		other_poses_ordered = other_poses_ordered[other_poses_ordered[:, 4].argsort()]
	for ag_id in range(min(n_other_agents,self.args.n_other_agents)):
		rel_pos = np.array([other_poses_ordered[ag_id,0] - current_pos[0],other_poses_ordered[ag_id, 1] - current_pos[1]])*\
							      multivariate_normal.pdf(np.linalg.norm(np.array([other_poses_ordered[ag_id,:2] - current_pos])),
							                                                  mean=0.0,cov=5.0)
		rel_vel = np.array([other_poses_ordered[ag_id,2] - current_vel[0],other_poses_ordered[ag_id, 3] - current_vel[1]])
		# pedestrian_grid[batch_idx, tbp_step, ag_id*4:ag_id*4+4] = np.concatenate([rel_pos, rel_vel])
		pedestrian_grid[ag_id*4:ag_id*4+4] = np.concatenate([rel_pos, rel_vel])
	return pedestrian_grid



# prev horizon
ped_1_obs_vel = [
[ 0.9945265,   0.06341162,  0.        ],
 [ 0.9945265,   0.06341175,  0.        ],
 [ 1.664607,   -0.09233275,  0.        ],
 [ 1.80472875,  0.247977,    0.        ],
 [ 1.35338,     0.037881,   0.        ],
 [ 1.72992985,  0.014407,   0.        ],
 [ 1.68241589, -0.2112955,   0.        ]]
#  [ 1.51948391 -0.3126315   0.        ],
#  [ 1.50622785 -0.19690975  0.        ],
#  [ 1.61588275 -0.0905315   0.        ],
#  [ 1.46069725 -0.19183     0.        ],
#  [ 1.55940425 -0.19564775  0.        ],
#  [ 1.66025025 -0.0911645   0.        ],
#  [ 1.6323035  -0.08963     0.        ],
#  [ 1.718679   -0.09437275  0.        ],
#  [ 1.5703155  -0.1917305   0.        ],
#  [ 1.6548695  -0.19455175  0.        ],
#  [ 1.626547   -0.1912225   0.        ],
#  [ 1.4766095  -0.490464    0.        ],
#  [ 1.536982   -0.38136925  0.        ],
#  [ 1.723151   -0.1843925   0.        ],
#  [ 1.69107775 -0.1809605   0.        ],
#  [ 1.655945   -0.27600525  0.        ],
#  [ 1.637185    0.021484    0.        ]]



# Trajectory class
# self.time_vec = time_vec    # timesteps in [ns]
# self.pose_vec = pose_vec    # [x, y, heading]
# self.vel_vec = vel_vec      # [vx, vy, omega]
# self.goal = goal            # [x,y]
# self.pose_interp = None
# self.vel_interp = None
# self.other_agents_positions = []  # store indices of other agents trajectories for each time step of this trajectory (with wich other positions does it need to be compared at a certain time)
# self.other_agents_velocities = []
    



# #create model and feed data
# model = NetworkModel(args)
# batch_x, batch_vel, batch_pos,batch_goal,batch_grid, batch_ped_grid, batch_y,batch_pos_target, other_agents_pos, new_epoch = data_prep.getBatch()

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


# dict = {"batch_x": batch_x,
# 			        "batch_vel": batch_vel,
# 			        "batch_pos": batch_pos,
# 			        "batch_grid": batch_grid,
# 			        "batch_ped_grid": other_agents_info,
# 			        "step": step,
# 			        "batch_goal": batch_goal,
# 			        "state_noise": 0.0,
# 			        "grid_noise": 0.0,
# 			        "concat_noise": 0.0,
# 			        "other_agents_pos": [other_agents_pos]
# 			        }
# feed_dict_ = model.feed_test_dic(**dict)

# y_model_pred, likelihood = model.predict(sess, feed_dict_, True)

# #all_predictions.append(predictions)

# processData

# trajectory_set
# agent_container
# traj = self.trajectory_set[trajectory_idx][1]
