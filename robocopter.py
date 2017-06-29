#!/usr/bin/env python
import os
import sys
import tensorflow as tf
from collections import deque
from tensorflow.contrib.layers import convolution2d, fully_connected
import cv2
from time import sleep

cam = cv2.VideoCapture('tcp://192.168.1.1:5555')
print('Video Stream Connected')

os.chdir('~/python-ardrone')
sys.path.insert(0, 'ardrone')
sys.path.insert(0, 'build/lib.linux-x86_64-3.6')

import ardrone

raw_width = 640
raw_height = 360
raw_channels = 3

def action_functions(act_no):
    if act_no == 0:
        drone.move_forward() 
    elif act_no == 1:
        drone.move_down() 
    elif act_no == 2:
        drone.move_up() 
    elif act_no == 3:
        drone.turn_left() 
    elif act_no == 4:
        drone.turn_right() 
    else:
        drone.move_up()

def prep_obs(obs):
    mean_color = obs.mean()
    img = obs[::4, ::4]
    img = img.mean(axis=2)
    img[img==mean_color] = 0
    img = ((img - 128) / 128)
    return img.reshape(90, 160, 1)


drone = ardrone.ARDrone()
drone.set_speed(0.05)    # Walk before you run
drone.reset()
print('Initialize Drone')
sleep(5)
drone.takeoff()
print('Drone Takeoff')
sleep(5)

running = True
i = 0
while running and i < 300:
    i += 1
    running, frame = cam.read()
    print(frame)
    print(drone.navdata)
drone.land()
cam.release()

# TODO: preprocess to the following
input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"]*3
conv_activation = [tf.nn.relu]*3
n_hidden_in = 64 * 11 * 10 # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n # 9 discrete actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()

def q_network(X_state, scope):
    prev_layer = X_state
    conv_layers = []
    with tf.variable_scope(scope) as scope:
	for n_maps, kernel_size, stride, padding, activation in zip(
	conv_n_maps, conv_kernel_sizes, conv_strides,
	conv_paddings, conv_activation):
	prev_layer = convolution2d(
	prev_layer, num_outputs=n_maps, kernel_size=kernel_size,
	stride=stride, padding=padding, activation_fn=activation,
	weights_initializer=initializer)
	conv_layers.append(prev_layer)
    last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
    hidden = fully_connected(
        last_conv_layer_flat, n_hidden, activation_fn=hidden_activation,
        weights_initializer=initializer)
    outputs = fully_connected(
        hidden, n_outputs, activation_fn=None,
        weights_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
actor_q_values, actor_vars = q_network(X_state, scope="q_networks/actor")
critic_q_values, critic_vars = q_network(X_state, scope="q_networks/critic")
copy_ops = [actor_var.assign(critic_vars[var_name]) for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, n_outputs), axis=1, keep_dims=True)

y = tf.placeholder(tf.float32, shape=[None, 1])
cost = tf.reduce_mean(tf.square(y - q_value))
global_step = tf.Variable(0, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost, global_step=global_step)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def sample_memories(batch_size):
    indices = rnd.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
	for idx in indices:
	    memory = replay_memory[idx]
	    for col, value in zip(cols, memory):
		col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))

eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if rnd.rand() < epsilon:
	return rnd.randint(n_outputs) # random action
    else:
	return np.argmax(q_values) # optimal action

n_steps = 100000 # total number of training steps
training_start = 1000 # start training after 1,000 game iterations
training_interval = 3 # run a training step every 3 game iterations
save_steps = 50 # save the model every 50 training steps
copy_steps = 25 # copy the critic to the actor every 25 training steps
discount_rate = 0.95
skip_start = 90 # skip the start of every game (it's just waiting time)
batch_size = 50
iteration = 0 # game iterations
checkpoint_path = "./robocopter_dqn.ckpt"
done = True # env needs to be reset

with tf.Session() as sess:
    if os.path.isfile(checkpoint_path):
	saver.restore(sess, checkpoint_path)
    else:
	init.run()
    while True:
	step = global_step.eval()
	if step >= n_steps:
	    break
	iteration += 1
	if done: # game over, start again
	    obs = env.reset()
	    for skip in range(skip_start): # skip the start of each game
		obs, reward, done, info = env.step(0)
	    state = preprocess_observation(obs)
        # Actor evaluates what to do
	q_values = actor_q_values.eval(feed_dict={X_state: [state]})
	action = epsilon_greedy(q_values, step)
        # Actor plays
	obs, reward, done, info = env.step(action)
	next_state = preprocess_observation(obs)
        # Let's memorize what just happened
	replay_memory.append((state, action, reward, next_state, 1.0 - done))
	state = next_state
	if iteration < training_start or iteration % training_interval != 0:
	    continue
        # Critic learns
	X_state_val, X_action_val, rewards, X_next_state_val, continues = (
	    sample_memories(batch_size))
	next_q_values = actor_q_values.eval(
	    feed_dict={X_state: X_next_state_val})
	max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
	y_val = rewards + continues * discount_rate * max_next_q_values
	training_op.run(feed_dict={X_state: X_state_val,
	                           X_action: X_action_val, y: y_val})
        # Regularly copy critic to actor
	if step % copy_steps == 0:
	    copy_critic_to_actor.run()
        # And save regularly
	if step % save_steps == 0:
	    saver.save(sess, checkpoint_path)
