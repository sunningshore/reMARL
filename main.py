import gymnasium as gym
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, MaxPooling2D, Dropout, GRU, GRUCell, \
    RNN, TimeDistributed, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import copy
import cv2 as cv
import itertools
from scipy.interpolate import interpn
from FIELD import FIELD
from utility import get_position, path_prediction
import time
from multi_uav_env import *
from embedded_PSO import *
from utilities import *
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=300)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# print(sys.argv)
if len(sys.argv) > 1:
    id_ = int(sys.argv[1])
    hyper_agent_num = int(sys.argv[2])  # Only takes odd numbers, one in the center.
    agent_num, obs_num = int(sys.argv[3]), int(sys.argv[4])
else:
    id_ = 0
    hyper_agent_num = 3  # Only takes odd numbers, one in the center.
    agent_num, obs_num = 1, 2

# Hyperparameters
swarm_center_x = 20
swarm_center_y = 150
obstacle_range = 10
guard = obstacle_range + 10

memory_depth = 1
if memory_depth == 1:
    state_size = (3+1, 5)  # env.observation_space.shape
    print("Size of State Space ->  {}".format(state_size))
else:
    state_size = (memory_depth, ) + (3, 5)  # env.observation_space.shape
    print("Size of State Space ->  {}".format(state_size))
action_size = (1, )  # env.action_space.shape[0]
print("Size of Action Space ->  {}".format(action_size))

upper_bound = np.pi / 4  # env.action_space.high[0]
lower_bound = -np.pi / 4  # env.action_space.low[0]

deltaS = 5
upper_angle = np.pi
upper_k = -np.pi/(1*deltaS)+np.pi/(1*deltaS)+np.pi/(1*deltaS)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def kmeans_distance(p1, p2):
    distance = np.sqrt(np.sum((p1 - p2) ** 2))
    # distance = np.dot(np.squeeze(p1), np.squeeze(p2)) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    return distance


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=128):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        self.buffer_counter_episode = 0
        self.pos_counter_episode = [0 for _ in range(hyper_agent_num)]

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, agent_num, ) + state_size)
        self.action_buffer = np.zeros((self.buffer_capacity, agent_num, ) + action_size)
        self.reward_buffer = np.zeros((self.buffer_capacity, agent_num))
        self.next_state_buffer = np.zeros((self.buffer_capacity, agent_num, ) + state_size)
        self.done_buffer = np.zeros((self.buffer_capacity, agent_num))

        self.state_buffer_tmp = []
        self.action_buffer_tmp = []
        self.reward_buffer_tmp = []
        self.next_state_buffer_tmp = []
        self.done_buffer_tmp = []

        self.feature_buffer = []
        self.feature_baseline_buffer = []
        self.state_buffer_episodic = []
        self.action_buffer_episodic = []
        self.reward_buffer_episodic = []
        self.next_state_buffer_episodic = []
        self.done_buffer_episodic = []

        self.pos_buffer = [[] for _ in range(hyper_agent_num)]  # agent [x, y] + obs [x, y] + center [x, y]
        self.pos_buffer_tmp = [[] for _ in range(hyper_agent_num)]
        self.pos_buffer_episodic = [[] for _ in range(hyper_agent_num)]

        self.id_buffer = np.zeros((self.buffer_capacity, 1))  # agent [x, y] + obs [x, y] + center [x, y]
        self.id_buffer_tmp = []
        self.id_buffer_episodic = []

        self.pos_id = 0

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

        self.state_buffer_tmp.append(obs_tuple[0])
        self.action_buffer_tmp.append(obs_tuple[1])
        self.reward_buffer_tmp.append(obs_tuple[2])
        self.next_state_buffer_tmp.append(obs_tuple[3])
        self.done_buffer_tmp.append(obs_tuple[4])

        # self.pos_buffer[obs_tuple[6]].append(obs_tuple[5])
        self.pos_id = obs_tuple[6]
        self.pos_buffer_tmp[self.pos_id].append(obs_tuple[5])

    # Takes (s,a,r,s') obervation tuple as input
    def record_episode(self):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        if self.buffer_counter_episode < self.buffer_capacity:
            # self.feature_buffer.append(feature)
            # self.feature_baseline_buffer.append(baseline)
            self.state_buffer_episodic.append(self.state_buffer_tmp)
            self.action_buffer_episodic.append(self.action_buffer_tmp)
            self.reward_buffer_episodic.append(self.reward_buffer_tmp)
            self.next_state_buffer_episodic.append(self.next_state_buffer_tmp)
            self.done_buffer_episodic.append(self.done_buffer_tmp)
            self.id_buffer_episodic.append(self.id_buffer_tmp)
        else:
            index = self.buffer_counter_episode % self.buffer_capacity

            # self.feature_buffer[index] = feature
            # self.feature_baseline_buffer[index] = baseline
            self.state_buffer_episodic[index] = self.state_buffer_tmp
            self.action_buffer_episodic[index] = self.action_buffer_tmp
            self.reward_buffer_episodic[index] = self.reward_buffer_tmp
            self.next_state_buffer_episodic[index] = self.next_state_buffer_tmp
            self.done_buffer_episodic[index] = self.done_buffer_tmp
            self.id_buffer_episodic[index] = self.id_buffer_tmp

        if self.pos_counter_episode[self.pos_id] < self.buffer_capacity:
            self.pos_buffer_episodic[self.pos_id].append(self.pos_buffer_tmp[self.pos_id])
        else:
            index = self.pos_counter_episode[self.pos_id] % self.buffer_capacity
            self.pos_buffer_episodic[self.pos_id][index] = self.pos_buffer_tmp[self.pos_id]

        self.pos_counter_episode[self.pos_id] += 1
        # self.pos_counter_episode += 1
        self.buffer_counter_episode += 1
        self.state_buffer_tmp = []
        self.action_buffer_tmp = []
        self.reward_buffer_tmp = []
        self.next_state_buffer_tmp = []
        self.done_buffer_tmp = []
        self.pos_buffer_tmp = [[] for _ in range(hyper_agent_num)]
        self.id_buffer_tmp = []

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    # @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape1:
            # tape1.watch(Critic1.trainable_variables)
            batch_len = next_state_batch.shape[0]
            actions_ = target_Actor(next_state_batch, training=True)
            critic_1 = target_Critic1(
                [next_state_batch, actions_], training=True
            )
            y1 = reward_batch + gamma * critic_1 * (1 - done_batch)
            critic1 = Critic1([state_batch, action_batch], training=True)
            critic_loss1 = huber_loss(y1, critic1)

        critic_grad = tape1.gradient(critic_loss1, Critic1.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, Critic1.trainable_variables)
        )
        with tf.GradientTape() as tape:
            actions = Actor(state_batch, training=True)
            critic = Critic1([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss1 = -tf.math.reduce_mean(critic)

        actor_grad = tape.gradient(actor_loss1, Actor.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, Actor.trainable_variables)
        )

        return actor_loss1, critic_loss1

    def update_(self, ):
        with tf.GradientTape() as tape:
            # Get sampling range
            record_range = min(self.buffer_counter, self.buffer_capacity)
            # Randomly sample indices
            batch_indices_swarm = []
            for _ in range(agent_num):
                batch_indices = np.random.choice(record_range, self.batch_size)
                batch_indices_swarm.append(batch_indices)

            # Convert to tensors
            state_batch = tf.convert_to_tensor(
                [np.stack(self.state_buffer[:, _])[batch_indices_swarm[_]] for _ in range(agent_num)])
            reward_batch = tf.convert_to_tensor(
                [np.stack(self.reward_buffer[:, _])[batch_indices_swarm[_]] for _ in range(agent_num)])
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            id_batch = tf.convert_to_tensor(
                [np.stack(self.id_buffer[:, _])[batch_indices_swarm[_]] for _ in range(agent_num)])

            state_batch = tf.reshape(state_batch, (-1,) + state_size)
            reward_batch = tf.reshape(reward_batch, (-1,) + (1,))
            id_batch = tf.reshape(id_batch, (-1,) + (1,))

            classifier_loss1 = tf.keras.losses.categorical_crossentropy(state_batch, id_batch, from_logits=False)

        classifier_grad = tape.gradient(classifier_loss1, Classifier.trainable_variables)
        classifier_optimizer.apply_gradients(
            zip(classifier_grad, Classifier.trainable_variables)
        )

        return actor_loss1, critic_loss1

    def quality_sampling(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices_swarm = []
        for _ in range(agent_num):
            batch_indices = np.random.choice(record_range, self.batch_size)
            batch_indices_swarm.append(batch_indices)
        # Convert to tensors
        state_batch = tf.convert_to_tensor([np.stack(self.state_buffer[:, _])[batch_indices_swarm[_]] for _ in range(agent_num)])
        action_batch = tf.convert_to_tensor([np.stack(self.action_buffer[:, _])[batch_indices_swarm[_]] for _ in range(agent_num)])
        reward_batch = tf.convert_to_tensor([np.stack(self.reward_buffer[:, _])[batch_indices_swarm[_]] for _ in range(agent_num)])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor([np.stack(self.next_state_buffer[:, _])[batch_indices_swarm[_]] for _ in range(agent_num)])
        done_batch = tf.convert_to_tensor([np.stack(self.done_buffer[:, _])[batch_indices_swarm[_]] for _ in range(agent_num)])
        done_batch = tf.cast(done_batch, dtype=tf.float32)

        state_batch = tf.reshape(state_batch, (-1,) + state_size)
        action_batch = tf.reshape(action_batch, (-1,) + action_size)
        reward_batch = tf.reshape(reward_batch, (-1,) + (1,))
        next_state_batch = tf.reshape(next_state_batch, (-1,) + state_size)
        done_batch = tf.reshape(done_batch, (-1,) + (1,))

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    # We compute the loss and update parameters
    def learn(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.quality_sampling()
        actor_loss, critic_loss = self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        return actor_loss, critic_loss


def actor_network(input_shape, optimizer):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)
    state_input = Input(input_shape)
    state_x = Flatten()(state_input)
    x = Dense(256, activation="relu")(state_x)
    angle = Dense(1, activation="tanh", kernel_initializer=last_init)(x)
    angle = angle * upper_bound
    model = Model(inputs=state_input, outputs=angle)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def critic_network(input_shape, action_shape, optimizer):
    state_input = Input(input_shape)
    state_x = Flatten()(state_input)
    state_x = Dense(256, activation="relu")(state_x)
    action_input = Input(action_shape)
    action_x = Flatten()(action_input)
    action_x = Dense(256, activation="relu")(action_x)
    out_x = Concatenate()([state_x, action_x])
    out_x = Dense(256, activation="relu")(out_x)
    value = Dense(1, kernel_initializer='he_uniform')(out_x)
    model = Model(inputs=[state_input, action_input], outputs=[value])
    model.compile(loss='mse', optimizer=optimizer)
    return model


def policy(state_, noise_object, explore_p_):
    if test:
        sampled_actions = [tf.squeeze(Actor[_](tf.expand_dims(state_[_], axis=0))) for _ in range(hyper_agent_num)]
        sampled_actions = np.stack(sampled_actions)
    else:
        sampled_actions = Actor(state_)
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise
    # We make sure action is within bounds
    legal_actions = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_actions)]


def vehicle_distances(env_, guard_, ite_swarm_main_):
    # UAVs
    vehicle_positions = []
    for idx, vehicle in enumerate(env_.agent):
        vehicle_positions.append((vehicle.x, vehicle.y))
    vehicle_positions = np.stack(vehicle_positions)
    u2o_distance = []
    obstacle_positions = []
    idx = 1
    for vehicle in env_.obstacle:
        u2o_dist_min = np.min(np.sqrt(np.sum((vehicle_positions - (vehicle.x, vehicle.y)) ** 2, axis=1)))
        u2o_distance.append(u2o_dist_min)
        obstacle_positions.append((vehicle.x, vehicle.y))
        idx += 1
    obstacle_positions = np.stack(obstacle_positions)
    u2o_distance = np.min(u2o_distance)

    if ite_swarm_main_ == 1:
        u2u_distance = guard_ + 10
    else:
        comb = itertools.combinations(np.arange(ite_swarm_main_), 2)
        comb = np.array(list(comb))
        u2u_distance = np.min(np.sqrt(np.sum((vehicle_positions[comb[:, 0], :] - vehicle_positions[comb[:, 1], :]) ** 2, axis=1)))

    return vehicle_positions, obstacle_positions, u2o_distance, u2u_distance


std_dev = .1
ou_noise = OUActionNoise(mean=np.zeros([agent_num, 1]), std_deviation=float(std_dev) * np.ones([agent_num, 1]))

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

Actor = actor_network(input_shape=state_size, optimizer=actor_optimizer)
target_Actor = actor_network(input_shape=state_size, optimizer=actor_optimizer)
Critic1 = critic_network(input_shape=state_size, action_shape=action_size, optimizer=critic_optimizer)

# Making the weights equal initially
target_Actor.set_weights(Actor.get_weights())
target_Critic1.set_weights(Critic1.get_weights())

total_episodes = 90_000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
batch_size = 128
buffer = Buffer(10000, batch_size)


# To store reward history of each episode
ep_reward_list = []
ep_reward_shadow_list = []
# To store average reward history of last few episodes
avg_reward_list = []

epsilon_min = 0
epsilon = 1
epsilon_decay = 1e-04
decay_step = 0

N = 10
swarmFieldRange, obsFieldRange = 100, 100
field_size = [300, 300]
field = FIELD(agent_num, obs_num, guard, field_size)
virtual_leader_offset = 10

# id_ = 0
algo_name = 'pendulum'
test = False
if test:
    Actor = [actor_network(input_shape=state_size, optimizer=actor_optimizer) for _ in range(hyper_agent_num)]
    for _ in range(hyper_agent_num):
        Actor[_].load_weights(algo_name + "_actor_" + str(hyper_agent_num) + 'u' + str(obs_num) + 'o_' + str(id_) + ".weights.h5")

    Critic = [critic_network(input_shape=state_size, action_shape=action_size, optimizer=critic_optimizer) for _ in range(hyper_agent_num)]
    for _ in range(hyper_agent_num):
        Critic[_].load_weights(algo_name + "_critic_" + str(hyper_agent_num) + 'u' + str(obs_num) + 'o_' + str(id_) + ".weights.h5")

env = multi_uav_env(hyper_agent_num, agent_num, obs_num, swarm_center_x, swarm_center_y, guard, test)

if test:
    ite_swarm_main = hyper_agent_num
else:
    ite_swarm_main = agent_num
best_return = 0
prev_actor_loss = 0
prev_critic_loss = 10000
actor_stop_counter = 0
critic_stop_counter = 0
early_stop_counter = 10000
early_stop_flag = False
time_lapsed = []
Ecurv_episodes = []
u2o_distance_episodes = []
u2u_distance_episodes = []
critic1set = []
for ep in range(total_episodes):
    if not test:
        if ep % 100 == 0:
            target_Actor.set_weights(Actor.get_weights())
            target_Critic1.set_weights(Critic1.get_weights())
    prev_state, img = env.reset()
    step = 0
    ######################################################
    Vehicle_positions, Obstacle_positions, U2O_distance, U2U_distance = vehicle_distances(env, guard, ite_swarm_main)
    arcxPre, arcyPre = np.zeros([N, 1, agent_num]), np.zeros([N, 1, agent_num])
    field.environment_field(env.virtual_position, Vehicle_positions, Obstacle_positions, swarmFieldRange, obsFieldRange, 0)
    field_env = field.field_env
    swarm_position_history = []
    for _ in range(10):
        swarm_position_history.append(Vehicle_positions - [20 * _, 0])
    swarm_position_history = np.transpose(np.stack(swarm_position_history), [1, 2, 0])
    ######################################################
    episodic_reward = 0
    episodic_reward_shadow = 0
    u2o_distance_episode = []
    u2u_distance_episode = []
    critic1episodic = []
    terminated, truncated = (False, False), False
    while not (any(terminated) or truncated):
        img = env.render()
        ######################################################
        Vehicle_positions, Obstacle_positions, U2O_distance, U2U_distance = vehicle_distances(env, guard, ite_swarm_main)
        u2o_distance_episode.append(U2O_distance)
        u2u_distance_episode.append(U2U_distance)

        tf_prev_state = tf.convert_to_tensor(prev_state)

        decay_step += 1
        explore_p = epsilon_min + (epsilon - epsilon_min) * np.exp(-epsilon_decay * decay_step)

        action = policy(tf_prev_state, ou_noise, explore_p)

        critic1 = Critic1([tf_prev_state, tf.transpose(tf.stack(action), [1, 0])], training=False)
        critic1episodic.append(critic1)

        step += 1
        action_swarm = tuple(np.concatenate([np.expand_dims(np.hstack(action), 1),
                                             np.zeros([ite_swarm_main, 1]) + 0], axis=1))

        # Recieve state and reward from environment.
        state, reward_hw, terminated, truncated, info = env.step(action_swarm)
        done = terminated
        ######################################################
        field.environment_field(env.virtual_position, Vehicle_positions, Obstacle_positions, swarmFieldRange, obsFieldRange, 0)
        field_env = field.field_env
        swarm_position_history = np.concatenate(
            (np.expand_dims(Vehicle_positions, axis=-1), swarm_position_history), axis=2)
        ######################################################
        # E2Coop rewards
        gamma1, gamma2 = 1, 1
        E2Coop_len = 5
        x_sd = swarm_position_history[:, 1, 0:E2Coop_len]
        y_sd = swarm_position_history[:, 0, 0:E2Coop_len]
        N = 100  # Path granularity
        x_traject = np.linspace(x_sd[:, 0], x_sd[:, 1], num=N)
        y_traject = np.linspace(y_sd[:, 0], y_sd[:, 1], num=N)
        for _ in range(1, E2Coop_len-1):
            x_traject = np.concatenate([x_traject, np.linspace(x_sd[:, _], x_sd[:, _+1], num=N)[1:, :]], axis=0)
            y_traject = np.concatenate([y_traject, np.linspace(y_sd[:, _], y_sd[:, _+1], num=N)[1:, :]], axis=0)
        Econt = 0
        x_t = np.gradient(x_traject[:, 0])
        y_t = np.gradient(y_traject[:, 0])
        vel = np.array([[x_t[_], y_t[_]] for _ in range(x_t.size)])
        speed = np.sqrt(x_t * x_t + y_t * y_t)
        tangent = np.array([1 / (speed + 1e-16)] * 2).transpose() * vel
        ss_t = np.gradient(speed)
        xx_t = np.gradient(x_t)
        yy_t = np.gradient(y_t)
        curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t + 1e-16) ** 1.5
        Ecurv = np.max(curvature_val)
        Eint = Econt + Ecurv
        points = (np.linspace(0, field_size[0] - 1, field_size[0]), np.linspace(0, field_size[1] - 1, field_size[1]))

        poses = []
        if ep > hyper_agent_num and not test:
            for _ in range(hyper_agent_num):
                if buffer.pos_counter_episode[_] < buffer.buffer_capacity:
                    index = buffer.pos_counter_episode[_]
                else:
                    index = buffer.pos_counter_episode[_] % buffer.buffer_capacity
                pos = buffer.pos_buffer_episodic[_][index - 1][-1][:2]
                poses.append(pos)

            comb = itertools.combinations(np.arange(hyper_agent_num), 2)
            comb = np.array(list(comb))
            u2u_distance = np.min(
                np.sqrt(np.sum((np.stack(poses)[comb[:, 0], :] - np.stack(poses)[comb[:, 1], :]) ** 2, axis=1)))
            if u2u_distance < guard:
                print('Embedded PSO here! ')
                # Embedded PSO
                dimensions = 2 * hyper_agent_num  # Let's only adjust y-axis, along the radius of the obstacle field.
                n_particles, iterations = 100, 30
                agent = PSO(dimensions, n_particles, iterations, np.hstack(poses), Obstacle_positions, guard)
                agent.search()
                poses = np.reshape(agent.pso_swarm.best_pos, [hyper_agent_num, -1])

        ext_field_set, Eext = [], np.zeros(
            [ite_swarm_main, 1])  # External field calculated based on the previous field_env and the previous swarm_position
        contours_swarm = []
        for _ in range(ite_swarm_main):
            ext_field = field_env.copy()
            ext_field_ = field_env.copy()
            if ep > hyper_agent_num and not test:
                point = poses[ep % hyper_agent_num]
            else:
                point = swarm_position_history[_, :, 1]

            point[point < 0] = 0
            point[point > field_size[0] - 1] = field_size[0] - 1
            field_value = interpn(points, field_env, point)
            ret, thresh = cv.threshold(field_env, field_value[0], 1.0, cv.THRESH_BINARY)
            contours, hierarchy = cv.findContours(thresh.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contours = np.squeeze(np.vstack(contours))
            contours_swarm.append(contours)
            ext_field[ext_field_ > field_value * 1.5] = -1
            ext_field[ext_field_ <= field_value * 0.9] = -1

            grad_field = np.gradient(ext_field)
            gamp = ext_field
            ext_field_set.append(gamp)
            ext_value = []
            for __ in range(N):
                point = np.array([x_traject[__, _], y_traject[__, _]])
                point[point < 0] = 0
                point[point > field_size[0] - 1] = field_size[0] - 1
                ext_value.append(interpn(points, gamp, np.flip(point)))
            Eext[_] = np.sum(ext_value) / (np.sqrt(2) * N)
        E = gamma1 * Eext - gamma2 * Eint.transpose()

        # plt.clf()
        # plt.xlim(0, 300)
        # plt.ylim(0, 300)
        # plt.imshow(np.zeros([300, 300, 3], int) + 255)
        # for _ in range(hyper_agent_num):
        #     plt.plot(x_traject[:, _], y_traject[:, _], 'r-')
        #     plt.plot(Vehicle_positions[_, 1], Vehicle_positions[_, 0], 'rs')
        #     plt.text(Vehicle_positions[_, 1] + 5, Vehicle_positions[_, 0] + 5, str(_), fontsize=12)
        #     plt.plot(env.virtual_position[1], env.virtual_position[0], 'ro')
        #     plt.plot(contours_swarm[_][:, 0], contours_swarm[_][:, 1], 'gray', linestyle='dashed')
        # for _ in range(obs_num):
        #     plt.plot(Obstacle_positions[_, 1], Obstacle_positions[_, 0], 'bs')
        # plt.axis('off')
        # plt.pause(0.1)
        # plt.show()
        # plt.savefig('figure_' + str(step) + '.pdf', format='pdf')

        if test:
            reward = sum(reward_hw)
        else:
            reward = (np.array(reward_hw) + np.squeeze(E))

        reward_shadow = np.sum(reward_hw)

        if not test:
            buffer.record((prev_state, np.array(action).transpose(), reward, state, np.array(done).transpose(), np.array([env.agent[0].x, env.agent[0].y, env.virtual_position[0], env.virtual_position[1]]), ep % hyper_agent_num))

        episodic_reward += np.mean(reward)
        episodic_reward_shadow += reward_shadow

        if not test and ep > 1:
            actor_loss_tf, critic_loss_tf = buffer.learn()
            if prev_actor_loss > actor_loss_tf.numpy():
                prev_actor_loss = actor_loss_tf.numpy()
            else:
                actor_stop_counter += 1
            if prev_critic_loss > critic_loss_tf.numpy():
                prev_critic_loss = critic_loss_tf.numpy()
            else:
                critic_stop_counter += 1

        prev_state = state

    if not test:
        buffer.record_episode()

    ep_reward_list.append(episodic_reward)
    ep_reward_shadow_list.append(episodic_reward_shadow)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    avg_reward_shadow = np.mean(ep_reward_shadow_list[-40:])
    print("Algo {} Run {} Episode {} Avg Reward {} --- prob: {}".format(algo_name, id_, ep, avg_reward, explore_p))
    avg_reward_list.append(avg_reward)

    if episodic_reward_shadow > best_return and not test:
        best_return = episodic_reward_shadow
        # Save the weights
        Actor.save_weights(algo_name + "_actor_" + str(hyper_agent_num) + 'u' + str(obs_num) + 'o_' + str(id_) + ".weights.h5")
        target_Actor.save_weights(algo_name + "_target_actor_" + str(hyper_agent_num) + 'u_' + str(obs_num) + 'o' + str(id_) + ".weights.h5")
        Critic1.save_weights(algo_name + "_critic_" + str(hyper_agent_num) + 'u' + str(obs_num) + 'o_' + str(id_) + ".weights.h5")
        target_Critic1.save_weights(algo_name + "_target_critic_" + str(hyper_agent_num) + 'u_' + str(obs_num) + 'o' + str(id_) + ".weights.h5")

    if not test:
        np.savetxt(algo_name + '_' + str(hyper_agent_num) + 'u' + str(obs_num) + 'o' + "_ep_reward_e2coop_" + str(id_) + ".csv", ep_reward_list, delimiter=", ", fmt='%f')
        np.savetxt(algo_name + '_' + str(hyper_agent_num) + 'u' + str(obs_num) + 'o' + "_ep_reward_" + str(id_) + ".csv", ep_reward_shadow_list, delimiter=", ", fmt='%f')
