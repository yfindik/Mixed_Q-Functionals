import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, seed=0):
        world = World()

        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True

        # add landmarks
        world.landmarks = [Landmark() for i in range(2)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        np.random.seed(seed)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
            agent.size = 0.2
            agent.is_done = False
            agent.collision_count = 0
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
        world.landmarks[0].color = np.array([0.75,0.25,0.25])
        world.landmarks[1].color = np.array([0.75, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        dists = np.sqrt([np.sum(np.square(agent.state.p_pos - lm.state.p_pos)) for lm in world.landmarks])
        _sum = sum(dists)
        _dist = _sum
        _min = min(dists)
        if _min <= 0.15:
            _dist = _min

        if _min <= 0.10:
            agent.is_done = True
        else:
            agent.is_done = False

        for i in world.agents:
            if i == agent:
                continue
            elif self.is_collision(agent, i):
                agent.collision_count += 1
        return np.exp(-(_dist ** 2) / 0.1)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)
