import lgsvl
import os
from math import sqrt
import time


class LgsvlEnv():

    def __init__(self):
        self.env = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)
        # загрузка сцены
        if self.env.current_scene == "BorregasAve":
            self.env.reset()
        else:
            self.env.load("BorregasAve")
        self.control = lgsvl.VehicleControl()
        self.vehicles = dict()

    def reset(self):
        self.done = False
        self.steps = 0
        self.env.reset()
        self.v = 0
        self.control.steering = 0

        state = lgsvl.AgentState()
        state.transform.position = lgsvl.Vector(0, -2, 0)
        state.transform.rotation.x = 0
        state.transform.rotation.y = 180
        state.transform.rotation.z = 0
        self.ego = self.env.add_agent(name="Lincoln2017MKZ (Apollo 5.0)", agent_type=lgsvl.AgentType.EGO, state = state)
        self.px = self.ego.transform.position.x
        self.py = self.ego.transform.position.y
        self.pz = self.ego.transform.position.z

        self.nx = self.ego.transform.position.x
        self.ny = self.ego.transform.position.y
        self.nz = self.ego.transform.position.z

        state.transform.position = lgsvl.Vector(40, -2, 0)
        npc = self.env.add_agent("Bob", lgsvl.AgentType.PEDESTRIAN, state=state)

        return self.get_observation()

    def get_observation(self):
        self.x = self.ego.transform.position.x
        self.y = self.ego.transform.position.y
        self.z = self.ego.transform.position.z
        self.rot = self.ego.transform.rotation.y
        self.steer = self.control.steering
        return [round(self.x, 1), round(self.y, 1), round(self.z, 1), round(self.rot, 1), round(self.steer, 1)]


    def step(self, action):
        self.info = {}
        self.reward = 0

        if action == 0 and self.control.steering >= -1:
            self.control.steering -= 0.1
        elif action == 1 and self.control.steering <= 1:
            self.control.steering += 0.1
        else:
            self.reward = -2

        if self.ego.state.speed <= 4:
            self.control.throttle = 1.0
            self.control.braking = 0.0
        elif self.ego.state.speed <= 5:
            self.control.throttle = 0.25
            self.control.braking = 0.0
        else:
            self.control.throttle = 0.0
            self.control.braking = 0.1


        self.ego.apply_control(self.control, sticky=True)

        self.env.run(0.1)
        self.steps += 1
        self.ego.on_collision(self._on_collision)
        if not self.done:
            self.reward = self.calculate_reward()

        self.px = self.x
        self.py = self.y
        self.pz = self.z

        return self.get_observation(), self.reward, self.done, self.info

    def calculate_reward(self):

        d = sqrt((self.x - 40) ** 2 + (self.z + 0) ** 2)

        if round(self.px, 0) == round(self.x, 0) and round(self.py, 0) == round(self.y, 0) and round(self.pz, 0) == round(self.z, 0):
            self.reward = -1
            self.v += 1
            if self.v == 10:
                self.reward = -200
                self.v = 0
                self.done = True
                return self.reward
            else:
                return self.reward
        elif d < 6:
            time.sleep(0.1)
            self.reward = 200
            self.v = 0
            self.done = True
            return self.reward
        elif d < 12:
            self.reward = 2
            self.v = 0
            return self.reward
        elif d < 18:
            self.reward = 1
            self.v = 0
            return self.reward
        elif d < 24:
            self.reward = 0.5
            self.v = 0
            return self.reward
        elif d < 30:
            self.reward = 0.1
            self.v = 0
            return self.reward
        elif d < 40:
            self.reward = -0.1
            self.v = 0
            return self.reward
        elif d < 50:
            self.reward = -0.4
            self.v = 0
            return self.reward
        elif d < 60:
            self.reward = -0.8
            self.v = 0
            return self.reward
        elif self.y < -3:
            self.reward = -200
            self.v = 0
            self.done = True
            return self.reward
        else:
            self.reward = -1
            self.v = 0
            return self.reward




    def _on_collision(self, agent1, agent2, contact):
        name1 = "STATIC OBSTACLE" if agent1 is None else agent1.name
        name2 = "STATIC OBSTACLE" if agent2 is None else agent2.name
        print("{} collided with {} at {}".format(name1, name2, contact))
        self.reward -= 200
        self.done = True





