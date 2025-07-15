import sys
import copy
import itertools
from pyswarms.backend.handlers import OptionsHandler
from pyswarms.backend.operators import compute_pbest
import pyswarms as ps
import pyswarms.backend as P
from pyswarms.backend.topology import Star
import matplotlib.pyplot as plt
import numpy as np


class PSO:
    def __init__(self, dimensions_, n_particles_, iterations_, init_position_, obs_position_, guard_):
        self.pso_options = {'c1': 0.6, 'c2': 0.3, 'w': 0.4}  # arbitrarily set
        self.dimensions = dimensions_
        self.n_particles = n_particles_

        # Initialize PSO particle position using current network parameters.
        self.init_position = init_position_
        self.obs_position = obs_position_
        self.agent_num = len(init_position_) / 2
        self.pso_swarm = P.create_swarm(n_particles=self.n_particles, dimensions=self.dimensions, options=self.pso_options)  # The Swarm Class
        self.pso_swarm.position = self.init_position + np.random.normal(loc=20, scale=10, size=[self.n_particles, self.dimensions])

        self.pso_topology = Star()  # The Topology Class
        self.iterations = iterations_  # Set 100 iterations
        self.guard = guard_

    def cost(self, positions_):
        positions = positions_
        cost = []
        for _ in range(self.n_particles):
            # position_x = np.reshape(self.init_position_x, [-1, 1])
            position = np.reshape(positions[_, :], [-1, 2])
            combs = itertools.combinations(np.arange(self.agent_num), 2)
            combs = np.array(list(combs), int)
            u2u_distances = np.sqrt(
                np.sum((position[combs[:, 0], :] - position[combs[:, 1], :]) ** 2, axis=1))
            cost1 = np.min(u2u_distances) - self.guard
            if cost1 <= 0:
                cost1 = 100

            u2o_distance = []
            for obs_position in self.obs_position:
                u2o_dist_min = np.min(np.sqrt(np.sum((position - obs_position) ** 2, axis=1)))
                u2o_distance.append(u2o_dist_min)
            cost2 = np.min(u2o_distance) - self.guard
            if cost2 <= 0:
                cost2 = 100

            cost3 = np.max((np.sqrt(np.sum((position - np.reshape(self.init_position, [-1, 2])) ** 2, axis=1))))

            cost0 = cost1 + cost3

            cost.append(cost0)

        return np.array(cost)

    def test(self):
        print('This file is modified. ')

    def search(self):
        # plt.figure(101)
        for _ in range(self.iterations):
            # Part 1: Update personal best
            self.pso_swarm.current_cost = self.cost(self.pso_swarm.position)  # Compute current cost
            self.pso_swarm.pbest_cost = self.cost(self.pso_swarm.pbest_pos)  # Compute personal best pos
            self.pso_swarm.pbest_pos, self.pso_swarm.pbest_cost = P.compute_pbest(self.pso_swarm)  # Update and store

            # Part 2: Update global best
            # Note that gbest computation is dependent on your topology
            if np.min(self.pso_swarm.pbest_cost) < self.pso_swarm.best_cost:
                self.pso_swarm.best_pos, self.pso_swarm.best_cost = self.pso_topology.compute_gbest(self.pso_swarm)

            # Let's print our output
            if _ % 10 == 0:
                print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(_ + 1, self.pso_swarm.best_cost))
                # print('----------------------------------------------------------------------------------------------')
                self.pso_swarm.position += np.random.normal(loc=10, scale=10, size=[self.n_particles, self.dimensions])

            # Part 3: Update position and velocity matrices
            # Note that position and velocity updates are dependent on your topology
            self.pso_swarm.velocity = self.pso_topology.compute_velocity(self.pso_swarm) * 1
            self.pso_swarm.position = self.pso_topology.compute_position(self.pso_swarm)

