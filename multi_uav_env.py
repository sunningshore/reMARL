import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
import os
import configparser

matplotlib.use('TkAgg')
# size_env = [300, 300]


class Object:
    def __init__(self, uid, name, x, y, heading, speed):
        self.id = uid
        self.name = name
        self.x = x
        self.y = y
        self.traject_x = [x]
        self.traject_y = [y]
        self.prev_x = x
        self.prev_y = y
        self.LENGTH = 5.0
        self.heading = heading
        self.original_x = x
        self.original_y = y
        self.original_heading = heading
        self.speed = speed
        self.observe_range = 100
        self.collision_range = 20
        self.max_heading_change = np.pi / 4
        self.neighbor = []
        self.crashed = False

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def collision_check(self):
        for veh in self.neighbor:
            dist = np.sqrt((self.x - veh.x) ** 2
                           + (self.y - veh.y) ** 2)
            if dist <= self.collision_range:
                self.crashed = True

    def action(self, dh, ds, dt):
        delta_f = dh
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        dx = self.speed * np.cos(self.heading + beta)
        dy = self.speed * np.sin(self.heading + beta)
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += dx * dt
        self.y += dy * dt

        self.traject_x.append(self.x)
        self.traject_y.append(self.y)

        self.heading += beta / (self.LENGTH / 2) * dt
        self.speed += ds * dt

        self.collision_check()


def lmap(v, x, y):
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


class multi_uav_env:
    def __init__(self, hyper_agent_num, agent_num, obs_num, xc, yc, guard, test):
        self.hyper_agent_num = hyper_agent_num  # Only takes odd numbers, one in the center.
        self.yc_shift = np.arange(self.hyper_agent_num) - (self.hyper_agent_num - 1) / 2
        self.agent_num = agent_num
        self.obs_num = obs_num
        self.xc, self.yc = xc, yc
        self.guard = guard
        self.virtual_position = np.array([10, yc]) + np.array([guard / 3, 0])
        self.obstacle_spawn_time = 0
        self.agent = []
        self.obstacle = []
        self.size_env = [600, 300]
        self.entry, self.property = 3+1, 5
        self.velocity_scale = 20
        self.dt = 1
        self.step_count = 0
        self.max_step_count = 40
        self.forward_speed_range = [-10, 10]
        self.config = {
            "on_track_reward": 0.3,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.3,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "in_formation_reward": 0.4
        }
        self.test = test

    def reset_obs(self, x0, y0, h0, s0):
        theta = np.linspace(-np.pi / 2, np.pi / 2, self.obs_num)
        self.obstacle = []
        for _ in range(self.obs_num):
            self.obstacle.append(Object(_, 'obs', x0 + 30 * np.cos(theta[_]), y0 + 30 * np.sin(theta[_]), h0, s0))

    def reset_swarm(self, x0, y0, h0, s0):
        self.agent = []
        if self.test:
            theta = np.linspace(-np.pi/2, np.pi/2, self.hyper_agent_num)
            for _ in range(self.hyper_agent_num):
                self.agent.append(Object(_, 'agent', x0 + 30 * np.cos(theta[_]), y0 + 30 * np.sin(theta[_]), h0, s0))
        else:
            theta = np.linspace(-np.pi/2, np.pi/2, self.hyper_agent_num)
            for _ in range(self.agent_num):
                self.agent.append(Object(_, 'agent', x0 + 30 * np.cos(theta[self.obstacle_spawn_time % self.hyper_agent_num]), y0 + 30 * np.sin(theta[self.obstacle_spawn_time % self.hyper_agent_num]), h0, s0))

    def reset(self):
        # xa, ya = 20, 150
        xa, ya, ha, sa = self.xc, self.yc, 0, 20
        xo, yo, ho, so = xa + 200, self.yc, np.pi, 5
        self.reset_swarm(xa, ya, ha, sa)
        self.reset_obs(xo, yo, ho, so)
        if self.test:
            for _ in range(self.hyper_agent_num):
                for __ in range(self.hyper_agent_num):
                    if self.agent[_].id == self.agent[__].id:
                        pass
                    else:
                        self.agent[_].neighbor.append(self.agent[__])
                for __ in range(self.obs_num):
                    self.agent[_].neighbor.append(self.obstacle[__])
        else:
            for _ in range(self.agent_num):
                for __ in range(self.agent_num):
                    if self.agent[_].id == self.agent[__].id:
                        pass
                    else:
                        self.agent[_].neighbor.append(self.agent[__])
                for __ in range(self.obs_num):
                    self.agent[_].neighbor.append(self.obstacle[__])

        self.virtual_position = np.array([10, self.yc]) + np.array([self.guard / 3, 0])

        self.obstacle_spawn_time += 1

        self.step_count = 0
        states, img = self.get_image()

        return states, img

    def render(self):
        states, img = self.get_image()
        ######################################################################
        img = img * 255
        img = cv2.resize(img, (self.size_env[0], self.size_env[1]), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Time_' + str(0), img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyWindow('Time_' + str(0))
            return

    def step(self, action):
        # Execute actions
        for _ in range(self.obs_num):
            self.obstacle[_].action(np.pi/4 * (-1) ** np.random.randint(2), 0, 1)
        if self.test:
            for _ in range(self.hyper_agent_num):
                self.agent[_].action(action[_][0], action[_][1], 1)
        else:
            for _ in range(self.agent_num):
                self.agent[_].action(action[_][0], action[_][1], 1)
        # Generate state_
        states, img = self.get_image()

        self.virtual_position += np.array([20, 0])

        # Generate reward
        rewards = []
        for _ in range(self.agent_num):
            forward_speed = self.agent[_].speed * np.cos(self.agent[_].heading)
            scaled_speed = lmap(forward_speed, self.forward_speed_range, [0, 1])
            if np.abs(self.agent[_].y - self.agent[_].original_y) <= 1:
                on_track_reward = True
            else:
                on_track_reward = False
            if self.agent[_].original_y - 30 < self.agent[_].y < self.agent[_].original_y + 30:
                in_formation_reward = True
            else:
                in_formation_reward = False
            reward_items = {
                "collision_reward": float(self.agent[_].crashed),
                "on_track_reward": float(on_track_reward),
                "high_speed_reward": np.clip(scaled_speed, 0, 1),
                "in_formation_reward": float(in_formation_reward)
            }
            reward = sum(self.config.get(name, 0) * reward for name, reward in reward_items.items())
            reward *= (1 - reward_items['collision_reward'])

            rewards.append(reward)

        # Generate info
        self.step_count += 1

        if self.step_count >= self.max_step_count:
            truncated = True
        else:
            truncated = False

        terminated = [agent.crashed or agent.x <= 1 or agent.x >= self.size_env[1] or agent.y <= 1 or agent.y >= self.size_env[0] for agent in self.agent]

        info = 'Excuse me? What info do you expect from such a low-cost environment developed by cheap labors? '

        return states, rewards, terminated, truncated, reward_items

    def get_image(self):
        # Local observations
        if self.test:
            agent_num = self.hyper_agent_num
        else:
            agent_num = self.agent_num
        states = np.zeros((agent_num, self.entry, self.property))
        for _ in range(agent_num):
            near_ind = 0
            states[_, near_ind, :] = [1, lmap(self.agent[_].x, [0, self.size_env[1]], [0, 1]),
                                      lmap(self.agent[_].y, [0, self.size_env[0]], [0, 1]),
                                      lmap(self.agent[_].speed * np.cos(self.agent[_].heading),
                                           [-self.velocity_scale, self.velocity_scale], [0, 1]),
                                      lmap(self.agent[_].speed * np.sin(self.agent[_].heading),
                                           [-self.velocity_scale, self.velocity_scale], [0, 1])]
            near_ind += 1
            states[_, near_ind, :] = [0, lmap(self.virtual_position[0], [0, self.size_env[1]], [0, 1]),
                                      lmap(self.virtual_position[1], [0, self.size_env[0]], [0, 1]),
                                      lmap(self.agent[_].original_x, [0, self.size_env[1]], [0, 1]),
                                      lmap(self.agent[_].original_y, [0, self.size_env[0]], [0, 1])]
            ###################################################################################
            ## Obstacles velocity in state have both directions and magnitudes.
            ###################################################################################
            for __ in range(self.obs_num):
                dist = np.sqrt((self.agent[_].x - self.obstacle[__].x) ** 2
                               + (self.agent[_].y - self.obstacle[__].y) ** 2)
                if dist <= self.agent[_].observe_range:
                    near_ind += 1
                    states[_, near_ind, :] = [0, lmap(self.obstacle[__].x, [0, self.size_env[1]], [0, 1]),
                                              lmap(self.obstacle[__].y, [0, self.size_env[0]], [0, 1]),
                                              lmap(self.obstacle[__].speed * np.cos(self.obstacle[__].heading),
                                                   [-self.velocity_scale, self.velocity_scale], [0, 1]),
                                              lmap(self.obstacle[__].speed * np.sin(self.obstacle[__].heading),
                                                   [-self.velocity_scale, self.velocity_scale], [0, 1])]
        # Output images
        img = np.ones([self.size_env[1], self.size_env[0], 3])
        for _ in range(agent_num):
            shift = 0
            if 0 < self.agent[_].y < self.size_env[1] and 0 < self.agent[_].x < self.size_env[0]:
                img[round(self.agent[_].y) - 5:round(self.agent[_].y) + 5, round(self.agent[_].x - shift) - 5:round(self.agent[_].x - shift) + 5, 1] = 0  # Drones first layers
                img[round(self.agent[_].y) - 5:round(self.agent[_].y) + 5, round(self.agent[_].x - shift) - 5:round(self.agent[_].x - shift) + 5, 2] = 0  # Drones first layers

        for _ in range(self.obs_num):
            shift = 0
            if 0 < self.obstacle[_].y < self.size_env[1] and 0 < self.obstacle[_].x < self.size_env[0]:
                img[round(self.obstacle[_].y) - 5:round(self.obstacle[_].y) + 5, round(self.obstacle[_].x - shift) - 5:round(self.obstacle[_].x - shift) + 5, 0] = 0  # Obstacle successive layers
                img[round(self.obstacle[_].y) - 5:round(self.obstacle[_].y) + 5, round(self.obstacle[_].x - shift) - 5:round(self.obstacle[_].x - shift) + 5, 1] = 0  # Obstacle successive layers

        return states, img
