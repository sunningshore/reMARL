import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm


class FIELD:
    def __init__(self, agent_num, obs_num, guard, field_size):
        self.agent_num = agent_num
        self.obs_num = obs_num
        self.guard = guard
        self.field_size = field_size
        self.virtual_position = []
        self.field_swarm, self.field_obstacle, self.field_env = np.zeros(self.field_size), np.zeros(self.field_size), np.zeros(self.field_size)
        self.field_x = np.reshape(np.arange(0, self.field_size[0]), [self.field_size[0], 1]) * np.ones([1, self.field_size[0]])
        self.field_y = np.ones([self.field_size[1], 1]) * np.reshape(np.arange(0, self.field_size[1]), [1, self.field_size[1]])
        self.breakFlag = False
        self.killing_intensity = np.zeros(obs_num)

    def swarm_field(self, swarmFieldRange):
        field_dist = np.sqrt((self.field_x - self.virtual_position[0]) ** 2 + (self.field_y - self.virtual_position[1]) ** 2)
        field_dist[field_dist < 5] = 5
        self.field_swarm = 1/(field_dist**2.5 + 1e-16)
        self.field_swarm[field_dist > swarmFieldRange] = 0

    def obstacle_field(self, obstacle_position, obsFieldRange):
        field_obstacle = []
        for i in range(self.obs_num):
            field_dist = np.sqrt((self.field_x - obstacle_position[i][0]) ** 2 + (self.field_y - obstacle_position[i][1]) ** 2)
            field_dist[field_dist < 5] = 5
            field_temp = 1/(field_dist**2 + 1e-16)
            field_temp[field_dist > obsFieldRange] = 0
            field_obstacle.append(field_temp)
            self.killing_intensity[i] = np.max(field_temp)
        self.field_obstacle = np.sum(np.stack(field_obstacle), axis=0)

    def environment_field(self, virtual_position, swarm_position, obstacle_position, swarmFieldRange, obsFieldRange, offset_y):
        self.virtual_position = virtual_position
        self.swarm_field(swarmFieldRange)
        self.obstacle_field(obstacle_position, obsFieldRange)
        self.field_env = self.field_swarm * 1 + self.field_obstacle
        self.field_env /= (np.max(self.field_env) + 1e-16)
        # self.virtual_position = np.mean(swarm_position, axis=0) + [self.guard, offset_y * 8]
        # self.virtual_position += [self.guard, 0]

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(Field_x, Field_y, Field_env, cmap=cm.coolwarm)
#
# plt.imshow(Field_env)
# for i in range(agent_num):
#     plt.plot(swarm_position[i][1], swarm_position[i][0], 'ro')
# for i in range(obs_num):
#     plt.plot(obstacle_position[i][1], obstacle_position[i][0], 'rs')