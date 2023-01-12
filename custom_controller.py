import numpy as np
from flight_controller import FlightController
from drone import Drone
from typing import Tuple
from math import log

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class CustomController(FlightController):

    def __init__(self):
        self.ky = 1.0
        self.kx = 0.5
        self.abs_pitch_delta = 0.1
        self.abs_thrust_delta = 0.3
        self.h = 1
        self.h1 = 1
        self.h2 = 1
        self.h3 = 1
        self.h4 = 1
        self.states = []
        self.actions = []
        self.rewards = []
        values= np.zeros(20,20,20)
        actions = (np.array([50,4,3,2,1]))
        values = [values,actions]
        self.value_function = values

    def value_calc(self, state):
       
        if (np.random.rand() > 0.9):
            return np.random.randint(0,4)
        else:   
            max = max(self.value_function[state[0]][state[1]][state[2]])
        
            for n in range(5):
                if(max == self.value_function[state[0]][state[1]][state[2]][n]):
                    i=n
                return i

    def get_max_simulation_steps(self):
        return 3000 # You can alter the amount of steps you want your program to run for here

    def a(self, x):
        i = x - self.h
        return (1 - sigmoid(i))

    def b(self, x):
        i = x + self.h
        return sigmoid(i)

    def reward(self, drone: Drone):

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y

        dist = np.sqrt(np.square(dx) + np.square(dy))
        velocity = np.sqrt(np.square(drone.velocity_x) + np.square(drone.velocity_y))

        if (dist<self.h1):
            ontarget= True
        else:
            ontarget = False


        if velocity < 0.05:
            velocity_multiplier = 0
        else:
            velocity_multiplier= clamp(-10(velocity-0.1)**2+1 ,0,1)

        if drone.pitch == 0:
            angle_multiplier = 1
        else:
            angle_multiplier= clamp(-log(30*abs(drone.pitch)) ,0,1)
        
        rotational_multiplier = clamp( -(drone.pitch_velocity/20)**2+1,0,1)

        multiplier =  velocity_multiplier*angle_multiplier* rotational_multiplier

        reward = -dist + ontarget(10*multiplier)
        return reward

    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y

        distance = np.sqrt(np.square(dx) + np.square(dy))
        velocity = np.sqrt(np.square(drone.velocity_x) + np.square(drone.velocity_y))

        thrust_adj = np.clip(dy * self.ky, -self.abs_thrust_delta, self.abs_thrust_delta)
        target_pitch = np.clip(dx * self.kx, -self.abs_pitch_delta, self.abs_pitch_delta)
        delta_pitch = target_pitch - drone.pitch


        if:        
            # Current
            thrust_left = np.clip(0.5 + thrust_adj + delta_pitch, 0.0, 1.0)
            thrust_right = np.clip(0.5 + thrust_adj - delta_pitch, 0.0, 1.0)
        elif:
            if -:
                # Left
                thrust_left = 1
                thrust_right = 0
            elif +:
                # Right
                thrust_left = 0
                thrust_right = 1
        elif: 
            # Both on
            thrust_left = 1
            thrust_right = 1
        elif:
            # Both off
            thrust_left = 0
            thrust_right = 0

        # State
        state = (drone.pitch_velocity, drone.pitch, dx, dy, velocity)
        self.states.append(state)

        # Action
        action = 1
        self.actions.append(action)

        # Reward
        S = (self.h1 * (-np.square(velocity - self.h2))) - (self.h3*(np.square(delta_pitch))) - (self.h4*(drone.pitch))
        reward = self.a(distance) + self.b(S)
        self.rewards.append(reward)

        # The default controller sets each propeller to a value of 0.5 0.5 to stay stationary.
        return (thrust_left, thrust_right)

    def reinforce() -> Tuple[float, float]:
        thrust_left = 0.5
        thrust_right = 0.5

        return (thrust_left, thrust_right)

    def train(self):
        """A self contained method designed to train parameters created in the initialiser."""

        epochs = 1
        # --- Code snipped provided for guidance only --- #
        for n in range(epochs):
            # 1) modify parameters
            states, actions, rewards = self.states, self.actions, self.rewards

            # 2) create a new drone simulation
            drone = self.init_drone()

            # 3) run simulation
            for t in range(self.get_max_simulation_steps()):
                drone.set_thrust(self.get_thrusts(drone))
                drone.step_simulation(self.get_time_interval())

            # 4) measure change in quality
            total_prev_rewards = np.cumsum(rewards)
            total_new_rewards = np.cumsum(self.rewards)
            # 5) update parameters according to algorithm

            if(total_prev_rewards < total_new_rewards):
                self.states = states
                self.actions = actions
                self.rewards = rewards


        pass