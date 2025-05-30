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

    # def action(self, k, omega, ds, dt):
    #     N = 100
    #     deltaS = 20
    #     x0 = self.x
    #     y0 = self.y
    #     # if omega < 0:
    #     #     omega += 2*np.pi
    #     x1 = deltaS * np.cos(omega) + x0
    #     y1 = deltaS * np.sin(omega) + y0
    #
    #     linSpcInd = np.expand_dims(np.linspace(0, 1, N), axis=1)
    #     alpha1 = -(np.pi/2 - omega)
    #     theta10 = np.pi + alpha1
    #     theta11 = theta10 - k * deltaS
    #     deltaTheta1 = np.ones([N, 1]) * theta11 + linSpcInd * (theta10 - theta11)
    #
    #     alpha2 = np.pi/2 + omega
    #     theta20 = -(np.pi - alpha2)
    #     theta21 = theta20 - k * deltaS
    #     deltaTheta2 = np.ones([N, 1]) * theta20 + linSpcInd * (theta21 - theta20)
    #
    #     alpha3 = -(np.pi/2 - omega)
    #     theta30 = -(np.pi - alpha3)
    #     theta31 = theta30 - k * deltaS
    #     deltaTheta3 = np.ones([N, 1]) * theta31 + linSpcInd * (theta30 - theta31)
    #
    #     alpha4 = -(3*np.pi/2 - omega)
    #     theta40 = np.pi + alpha4
    #     theta41 = theta40 - k * deltaS
    #     deltaTheta4 = np.ones([N, 1]) * theta40 + linSpcInd * (theta41 - theta40)
    #
    #     alpha5 = -(np.pi/2 - omega)
    #     theta50 = -(np.pi - alpha5)
    #     theta51 = theta50 - k * deltaS
    #     deltaTheta5 = np.ones([N, 1]) * theta51 + linSpcInd * (theta50 - theta51)
    #
    #     alpha6 = -(3*np.pi/2 - omega)
    #     theta60 = (np.pi + alpha6)
    #     theta61 = theta60 - k * deltaS
    #     deltaTheta6 = np.ones([N, 1]) * theta60 + linSpcInd * (theta61 - theta60)
    #
    #     alpha7 = -(5*np.pi/2 - omega)
    #     theta70 = (np.pi + alpha7)
    #     theta71 = theta70 - k * deltaS
    #     deltaTheta7 = np.ones([N, 1]) * theta71 + linSpcInd * (theta70 - theta71)
    #
    #     alpha8 = -(3*np.pi/2 - omega)
    #     theta80 = -(np.pi - alpha8)
    #     theta81 = theta80 - k * deltaS
    #     deltaTheta8 = np.ones([N, 1]) * theta80 + linSpcInd * (theta81 - theta80)
    #
    #     xTemp = np.ones([N, 1]) * x0 + linSpcInd * (x1 - x0)
    #     yTemp = np.ones([N, 1]) * y0 + linSpcInd * (y1 - y0)
    #
    #     if k > 0 and omega <= np.pi/2:
    #         x = np.ones([N, 1]) * np.abs(1 / k) * np.cos(deltaTheta1) + np.ones([N, 1]) * (np.abs(1 / k) * np.cos(alpha1)) + np.ones([N, 1]) * x0
    #         y = np.ones([N, 1]) * np.abs(1 / k) * np.sin(deltaTheta1) + np.ones([N, 1]) * (np.abs(1 / k) * np.sin(alpha1)) + np.ones([N, 1]) * x0
    #
    #     elif k < 0 and omega <= np.pi / 2:
    #         x = np.ones([N, 1]) * np.abs(1 / k) * np.cos(deltaTheta2) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.cos(alpha2)) + np.ones([N, 1]) * x0
    #         y = np.ones([N, 1]) * np.abs(1 / k) * np.sin(deltaTheta2) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.sin(alpha2)) + np.ones([N, 1]) * x0
    #
    #     elif np.pi / 2 < omega <= np.pi and k > 0:
    #         x = np.ones([N, 1]) * np.abs(1 / k) * np.cos(deltaTheta3) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.cos(alpha3)) + np.ones([N, 1]) * x0
    #         y = np.ones([N, 1]) * np.abs(1 / k) * np.sin(deltaTheta3) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.sin(alpha3)) + np.ones([N, 1]) * x0
    #
    #     elif np.pi / 2 < omega <= np.pi and k < 0:
    #         x = np.ones([N, 1]) * np.abs(1 / k) * np.cos(deltaTheta4) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.cos(alpha4)) + np.ones([N, 1]) * x0
    #         y = np.ones([N, 1]) * np.abs(1 / k) * np.sin(deltaTheta4) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.sin(alpha4)) + np.ones([N, 1]) * x0
    #
    #     elif np.pi < omega <= 3*np.pi/2 and k > 0:
    #         x = np.ones([N, 1]) * np.abs(1 / k) * np.cos(deltaTheta5) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.cos(alpha5)) + np.ones([N, 1]) * x0
    #         y = np.ones([N, 1]) * np.abs(1 / k) * np.sin(deltaTheta5) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.sin(alpha5)) + np.ones([N, 1]) * x0
    #
    #     elif np.pi < omega <= 3*np.pi/2 and k < 0:
    #         x = np.ones([N, 1]) * np.abs(1 / k) * np.cos(deltaTheta6) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.cos(alpha6)) + np.ones([N, 1]) * x0
    #         y = np.ones([N, 1]) * np.abs(1 / k) * np.sin(deltaTheta6) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.sin(alpha6)) + np.ones([N, 1]) * x0
    #
    #     elif 3*np.pi/2 < omega <= 2*np.pi and k > 0:
    #         x = np.ones([N, 1]) * np.abs(1 / k) * np.cos(deltaTheta7) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.cos(alpha7)) + np.ones([N, 1]) * x0
    #         y = np.ones([N, 1]) * np.abs(1 / k) * np.sin(deltaTheta7) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.sin(alpha7)) + np.ones([N, 1]) * x0
    #
    #     elif 3*np.pi/2 < omega <= 2*np.pi and k < 0:
    #         x = np.ones([N, 1]) * np.abs(1 / k) * np.cos(deltaTheta8) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.cos(alpha8)) + np.ones([N, 1]) * x0
    #         y = np.ones([N, 1]) * np.abs(1 / k) * np.sin(deltaTheta8) + np.ones([N, 1]) * (
    #                     np.abs(1 / k) * np.sin(alpha8)) + np.ones([N, 1]) * x0
    #
    #     else:
    #         x = xTemp
    #         y = yTemp
    #
    #     self.x = x[int(N/2)][0]
    #     self.y = y[int(N/2)][0]
    #     self.speed += ds * dt
    #     self.heading = np.arctan((y0-self.y)/(x0-self.x))
    #     self.traject_x = x
    #     self.traject_y = y
    #     self.collision_check()

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

        # lanes = [120, 150, 180]
        # dy = [abs(self.y - lane) for lane in lanes]
        # if self.y < lanes[0]-15 or self.y > lanes[-1] + 15:
        #     pass
        # else:
        #     dy_argmin = np.argmin(dy)
        #     self.y = lanes[dy_argmin]

        # self.x = np.min([self.x, size_env[0]])
        # self.x = np.max([self.x, 0])
        # self.y = np.min([self.y, size_env[1]])
        # self.y = np.max([self.y, 1])

        # self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.heading += beta / (self.LENGTH / 2) * dt
        self.speed += ds * dt

        # self.heading = self.original_heading

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
        # self.config = {
        #     "collision_reward": -10,  # The reward received when colliding with a vehicle.
        #     "on_track_reward": 0.5,  # The reward received when driving on the right-most lanes, linearly mapped to
        #     # zero for other lanes.
        #     "high_speed_reward": 0.5,  # The reward received when driving at full speed, linearly mapped to zero for
        #     # lower speeds according to config["reward_speed_range"].
        # }
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
            # self.obstacle.append(Object(_, 'obs', x0, y0, h0, s0))

    def reset_swarm(self, x0, y0, h0, s0):
        self.agent = []
        if self.test:
            theta = np.linspace(-np.pi/2, np.pi/2, self.hyper_agent_num)
            for _ in range(self.hyper_agent_num):
                self.agent.append(Object(_, 'agent', x0 + 30 * np.cos(theta[_]), y0 + 30 * np.sin(theta[_]), h0, s0))
                # self.agent.append(Object(_, 'agent', x0, y0 + 20 * self.yc_shift[_], h0, s0))
        else:
            theta = np.linspace(-np.pi/2, np.pi/2, self.hyper_agent_num)
            for _ in range(self.agent_num):
                self.agent.append(Object(_, 'agent', x0 + 30 * np.cos(theta[self.obstacle_spawn_time % self.hyper_agent_num]), y0 + 30 * np.sin(theta[self.obstacle_spawn_time % self.hyper_agent_num]), h0, s0))

            # for _ in range(self.agent_num):
            #     self.agent.append(Object(_, 'agent', x0 + 30 * np.cos(theta[_]), y0 + 30 * np.sin(theta[_]), h0, s0))

    def reset(self):
        # xa, ya = 20, 150
        xa, ya, ha, sa = self.xc, self.yc, 0, 20
        # xa, ya, ha, sa = self.xc, self.yc + 20 * (-1) ** (self.obstacle_spawn_time % 2), 0, 20
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
        ######################################################################
        # fig = plt.figure(0)
        # plt.clf()
        # plt.xlim(0, 300)
        # plt.ylim(0, 300)
        # plt.imshow(np.zeros([self.size_env[0], self.size_env[1], 3], int) + 255)
        # for _ in range(self.obs_num):
        #     plt.plot(self.obstacle[_].x, self.obstacle[_].y, 'bs')
        # for _ in range(self.hyper_agent_num):
        #     plt.plot(self.agent[_].x, self.agent[_].y, 'rs')
        #     plt.plot(self.agent[_].traject_x, self.agent[_].traject_y, 'r')
        # plt.savefig('figure_' + str(self.step_count) + '.pdf', format='pdf')

        # for t in range(int(1/self.dt)):
        #     for _ in range(self.obs_num):
        #         vec = [self.obstacle[_].x-self.obstacle[_].prev_x, self.obstacle[_].y-self.obstacle[_].prev_y]
        #         plt.plot(self.obstacle[_].prev_x + vec[0] * self.dt, self.obstacle[_].prev_y + vec[1] * self.dt, 'ro')
        #     for _ in range(self.agent_num):
        #         vec = [self.agent[_].x - self.agent[_].prev_x, self.agent[_].y - self.agent[_].prev_y]
        #         plt.plot(self.agent[_].prev_x + vec[0] * self.dt, self.agent[_].prev_y + vec[1] * self.dt, 'bo')

            # print('render {}. '.format(t))

        # plt.pause(.01)
        ######################################################################

    def step(self, action):
        # Execute actions
        for _ in range(self.obs_num):
            # if np.random.randn() > 0.5 and self.step_count % 3 == 0:
            #     self.obstacle[_].action(np.pi/4 * (-1) ** np.random.randint(2), 0, 1)
            # else:
            #     self.obstacle[_].action(0, 0, 1)
            # # self.obstacle[_].action(0, np.pi, 1, 1)
            self.obstacle[_].action(0, 0, 1)
        if self.test:
            for _ in range(self.hyper_agent_num):
                # self.agent[_].action(action[_][0], action[_][1], 0, 1)
                self.agent[_].action(action[_][0], action[_][1], 1)
        else:
            for _ in range(self.agent_num):
                # self.agent[_].action(action[_][0], action[_][1], 0, 1)
                self.agent[_].action(action[_][0], action[_][1], 1)
        # Generate state_
        states, img = self.get_image()

        self.virtual_position += np.array([20, 0])

        # Generate reward
        rewards = []
        for _ in range(self.agent_num):
            forward_speed = self.agent[_].speed * np.cos(self.agent[_].heading)
            scaled_speed = lmap(forward_speed, self.forward_speed_range, [0, 1])
            # offset = np.abs(self.agent[_].y - self.agent[_].original_y)
            # scaled_offset = lmap(offset, [0, self.size_env[1] / 2], [0, 1])
            if np.abs(self.agent[_].y - self.agent[_].original_y) <= 1:
                on_track_reward = True
            else:
                on_track_reward = False
            # on_track_reward = 1 / (np.abs(self.agent[_].y - self.agent[_].original_y) + 1)
            # on_track_reward = lmap(on_track_reward, [1 / (self.size_env[0] + 1), 1], [0, 1])
            if self.agent[_].original_y - 30 < self.agent[_].y < self.agent[_].original_y + 30:
                in_formation_reward = True
            else:
                in_formation_reward = False
            # reward_items = {
            #     "collision_reward": float(self.agent[_].crashed),
            #     "on_track_reward": float(on_track_reward),
            #     "high_speed_reward": np.clip(scaled_speed, 0, 1),
            #     "in_formation_reward": float(in_formation_reward)
            # }
            # reward = sum(self.config.get(name, 0) * reward for name, reward in reward_items.items())
            # reward = lmap(reward,
            #               [self.config['collision_reward'], self.config['on_track_reward'] + self.config['high_speed_reward']],
            #               [-1, 1])
            # reward *= reward_items['in_formation_reward']

            reward_items = {
                "collision_reward": float(self.agent[_].crashed),
                "on_track_reward": float(on_track_reward),
                "high_speed_reward": np.clip(scaled_speed, 0, 1),
                "in_formation_reward": float(in_formation_reward)
            }
            reward = sum(self.config.get(name, 0) * reward for name, reward in reward_items.items())
            reward *= (1 - reward_items['collision_reward'])

            # reward = .1
            rewards.append(reward)

        # Generate info
        self.step_count += 1

        # if self.agent[0].x > self.obstacle[0].x:
        #     self.obstacle_spawn_time += 1
        #     xo, yo, ho, so = self.agent[0].x + 200, self.agent[0].y + 20 * (-1) ** (self.obstacle_spawn_time % 2), np.pi, 20
        #     self.reset_obs(xo, yo, ho, so)

        if self.step_count >= self.max_step_count:
            truncated = True
        else:
            truncated = False

        terminated = [agent.crashed or agent.x <= 1 or agent.x >= self.size_env[1] or agent.y <= 1 or agent.y >= self.size_env[0] for agent in self.agent]

        info = 'Excuse me? What info do you expect from such a low-cost environment developed by cheap labors? '

        # print('step {}. '.format(self.step_count))

        return states, rewards, terminated, truncated, reward_items

    # def get_state(self):
    #     # Local observations
    #     states = np.zeros((self.agent_num, self.entry, self.property))
    #     for _ in range(self.agent_num):
    #         states[_, 0, :] = [1, lmap(self.agent[_].x, [0, self.size_env[1]], [0, 1]),
    #                            lmap(self.agent[_].y, [0, self.size_env[0]], [0, 1]),
    #                            lmap(self.agent[_].speed * np.cos(self.agent[_].heading),
    #                                 [-self.velocity_scale, self.velocity_scale], [0, 1]),
    #                            lmap(self.agent[_].speed * np.sin(self.agent[_].heading),
    #                                 [-self.velocity_scale, self.velocity_scale], [0, 1])]
    #         ###################################################################################
    #         ## Obstacles velocity in state have both directions and magnitudes.
    #         ###################################################################################
    #         for __ in range(self.obs_num):
    #             dist = np.sqrt((self.agent[_].x - self.obstacle[__].x) ** 2
    #                            + (self.agent[_].y - self.obstacle[__].y) ** 2)
    #             if dist <= self.agent[_].observe_range:
    #                 states[_, 1, :] = [0, lmap(self.obstacle[__].x, [0, self.size_env[1]], [0, 1]),
    #                                    lmap(self.obstacle[__].y, [0, self.size_env[0]], [0, 1]),
    #                                    lmap(self.obstacle[__].speed * np.cos(self.obstacle[__].heading),
    #                                         [-self.velocity_scale, self.velocity_scale], [0, 1]),
    #                                    lmap(self.obstacle[__].speed * np.sin(self.obstacle[__].heading),
    #                                         [-self.velocity_scale, self.velocity_scale], [0, 1])]
    #     # Output images
    #     img = np.ones([self.size_env[1], self.size_env[0], 3])
    #     for _ in range(self.agent_num):
    #         shift = 0
    #         if 0 < self.agent[_].y < self.size_env[1] and 0 < self.agent[_].x < self.size_env[0]:
    #             img[round(self.agent[_].y) - 5:round(self.agent[_].y) + 5, round(self.agent[_].x - shift) - 5:round(self.agent[_].x - shift) + 5, 1] = 0  # Drones first layers
    #             img[round(self.agent[_].y) - 5:round(self.agent[_].y) + 5, round(self.agent[_].x - shift) - 5:round(self.agent[_].x - shift) + 5, 2] = 0  # Drones first layers
    #
    #     for _ in range(self.obs_num):
    #         shift = 0
    #         if 0 < self.obstacle[_].y < self.size_env[1] and 0 < self.obstacle[_].x < self.size_env[0]:
    #             img[round(self.obstacle[_].y) - 5:round(self.obstacle[_].y) + 5, round(self.obstacle[_].x - shift) - 5:round(self.obstacle[_].x - shift) + 5, 0] = 0  # Obstacle successive layers
    #             img[round(self.obstacle[_].y) - 5:round(self.obstacle[_].y) + 5, round(self.obstacle[_].x - shift) - 5:round(self.obstacle[_].x - shift) + 5, 1] = 0  # Obstacle successive layers
    #
    #     return states, img

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
            ###################################################################################
            ## UAVs (ego and neighbour) velocity in state only have directions.
            ###################################################################################
            # for __ in range(self.agent_num):
            #     if self.agent[_].id == self.agent[__].id:
            #         pass
            #     else:
            #         dist = np.sqrt((self.agent[_].x - self.agent[__].x) ** 2
            #                        + (self.agent[_].y - self.agent[__].y) ** 2)
            #         if dist <= self.agent[_].observe_range:
            #             near_ind += 1
            #             states[_, near_ind, :] = [0, lmap(self.agent[__].x, [0, self.size_env[1]], [0, 1]),
            #                                       lmap(self.agent[__].y, [0, self.size_env[0]], [0, 1]),
            #                                       lmap(self.agent[__].speed * np.cos(self.agent[__].heading),
            #                                            [-self.velocity_scale, self.velocity_scale], [0, 1]),
            #                                       lmap(self.agent[__].speed * np.sin(self.agent[__].heading),
            #                                            [-self.velocity_scale, self.velocity_scale], [0, 1])]
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
            # UAV Radar range
            # r = self.agent[_].observe_range
            # x_cent = round(self.agent[_].x)
            # y_cent = round(self.agent[_].y)
            # for angle in np.linspace(-np.pi, np.pi, 10):
            #     tp_x = int(np.ceil(r * np.cos(angle) + x_cent))
            #     tp_y = int(np.ceil(r * np.sin(angle) + y_cent))
            #     img[round(tp_y) - 1: round(tp_y) + 1, round(tp_x) - 1: round(tp_x) + 1, 1] = 0  # V2V range first layers
            #     img[round(tp_y) - 1: round(tp_y) + 1, round(tp_x) - 1: round(tp_x) + 1, 2] = 0  # V2V range first layers

        for _ in range(self.obs_num):
            shift = 0
            if 0 < self.obstacle[_].y < self.size_env[1] and 0 < self.obstacle[_].x < self.size_env[0]:
                img[round(self.obstacle[_].y) - 5:round(self.obstacle[_].y) + 5, round(self.obstacle[_].x - shift) - 5:round(self.obstacle[_].x - shift) + 5, 0] = 0  # Obstacle successive layers
                img[round(self.obstacle[_].y) - 5:round(self.obstacle[_].y) + 5, round(self.obstacle[_].x - shift) - 5:round(self.obstacle[_].x - shift) + 5, 1] = 0  # Obstacle successive layers

        return states, img
