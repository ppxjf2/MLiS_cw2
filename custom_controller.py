import numpy as np
from flight_controller import FlightController
from drone import Drone
from typing import Tuple

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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

    def get_max_simulation_steps(self):
        return 3000 # You can alter the amount of steps you want your program to run for here

    def a(self, x):
        i = x - self.h
        return (1 - sigmoid(i))

    def b(self, x):
        i = x + self.h
        return sigmoid(i)

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
            # Left
            thrust_left = 1
            thrust_right = 0
        elif:
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