import datetime
from math import floor, log
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drone import Drone
from flight_controller import FlightController


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

pos_actions=6

class CustomController(FlightController):

    def __init__(self):
        self.G = []
        self.ky = 1.0
        self.kx = 0.8
        self.abs_pitch_delta = 0.1
        self.abs_thrust_delta = 0.3
        self.h1 = 0.11
        self.states = []
        self.actions = []
        self.rewards = []
        self.action = 0
        self.nexttarget = [0,0]
        self.counter = 1

        values= np.random.random([20,40,30,10,pos_actions]); 
        values+= 100000
        #actions = (np.array([1,-1,-2,-3,-4]))
        #values = [values,actions]
        self.value_function = values

    def value_calc(self, state):
        if (np.random.rand() > 0.99):
            return np.random.randint(pos_actions)
        else:   
            max1 = max(self.value_function[state[0]][state[1]][state[2]][state[3]])
            i = 0
            for n in range(pos_actions):
                if(max1 == self.value_function[state[0]][state[1]][state[2]][state[3]][n]):
                    i=n
            return i

    def get_max_simulation_steps(self):
        return 3000 # You can alter the amount of steps you want your program to run for here

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

        if abs(drone.velocity_x) < 0.1 :
            velocity_multiplier = 0
        else:
            velocity_multiplier= clamp(-2*(velocity-0.1)**2+1 ,0.3,1)

        if drone.pitch == 0:
            angle_multiplier = 0.2
        else:
            angle_multiplier= clamp(-0.8*log(abs(drone.pitch)) ,0.1,1)
        
        #rotational_multiplier = clamp( -(drone.pitch_velocity/20)**2+1,0.1,1)
        multiplier =  angle_multiplier*angle_multiplier * velocity_multiplier
        wrongwaypunish = np.sign(dx*drone.pitch)
        wrongwaymulti = clamp(abs(drone.pitch),0,1)
        angle_punishment = 25*floor(abs(drone.pitch))
        #print(drone.pitch)
        reward = -3*(dist) + 0.1*abs(drone.velocity_x) - angle_punishment + 2 * self.counter**3 # - wrongwaymulti*wrongwaypunish
        #print (multiplier)
        #print(ontarget)
        if (self.nexttarget != drone.get_next_target()):
            self.nexttarget = drone.get_next_target()
            self.counter += 1
            reward += 2000 #+ #500*multiplier
        return reward

    def action_selection(self, drone: Drone) -> Tuple[float, float]:
        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y

        if(self.action == 0):        
            if (np.sign(dx) == np.sign(drone.pitch)):
                thrust_left = 0.5
                thrust_right = 0.5
            else:
                thrust_left = np.clip((np.sign(dx)/2)+1, 0.0, 1.0)
                thrust_right =np.clip(-((np.sign(dx)/2)+1), 0.0, 1.0)
                
        elif(self.action == 1):
            thrust_adj = np.clip(dy * self.ky, -self.abs_thrust_delta, self.abs_thrust_delta)
            target_pitch = np.clip(dx * (self.kx), -self.abs_pitch_delta, self.abs_pitch_delta)
            delta_pitch = (target_pitch - drone.pitch)*2

            thrust_left = np.clip(0.4 + thrust_adj + delta_pitch, 0.0, 1.0)
            thrust_right = np.clip(0.4 + thrust_adj - delta_pitch, 0.0, 1.0)
        elif(self.action == 2):

            isrightspin = np.sign(drone.pitch_velocity)
            thrust_left = 0.5 - 0.4* isrightspin
            thrust_right = 0.5 + 0.4* isrightspin
            

        elif(self.action == 3):
            if (np.sign(drone.pitch_velocity)== np.sign(dx)):
                thrust_left = np.clip(np.sign(dx), 0.4, 0.6)
                thrust_right = np.clip(np.sign(dx), 0.4, 0.6)
            else:
                thrust_left = np.clip(np.sign(dx), 0.2, 0.8)
                thrust_right = np.clip(np.sign(dx), 0.2, 0.8)

        elif(self.action == 4):
            thrust_adj = np.clip(dy * self.ky, -self.abs_thrust_delta, self.abs_thrust_delta)
            target_pitch = np.clip(dx * self.kx, -self.abs_pitch_delta, self.abs_pitch_delta)
            delta_pitch = target_pitch - drone.pitch

            thrust_left = np.clip(0.5 + thrust_adj + delta_pitch, 0.0, 1.0)
            thrust_right = np.clip(0.5 + thrust_adj - delta_pitch, 0.0, 1.0)

        elif(self.action == 5):    
            
            thrust_left = np.clip(np.sign(dy), 0.0, 1.0)
            thrust_right = np.clip(np.sign(dy), 0.0, 1.0)
        
        return (thrust_left, thrust_right)

    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y

        # State
        state = (
            int(((drone.pitch-1)*10)%20),
            int(((dx-1.5)*8)%40),
            int(((dy-1.5)*10)%30),
            int(np.sqrt(np.square(drone.velocity_x) + np.square(drone.velocity_y))*10)%10
        )
        self.states.append(state)

        # Action
        self.action = self.value_calc(state)
        self.actions.append(self.action)

        thrust_left, thrust_right = self.action_selection(drone)

        # Reward
        reward = self.reward(drone)
        self.rewards.append(reward)

        # The default controller sets each propeller to a value of 0.5 0.5 to stay stationary.
        return (thrust_left, thrust_right)

    def train(self):
        """A self contained method designed to train parameters created in the initialiser."""
        self.values= np.random.random([20,40,30,10,pos_actions]); 
        self.values+= 100000
        epochs = 1
        average_returns = np.empty([20,40,30,10,pos_actions])
        average_returns_count = np.zeros([20,40,30,10,pos_actions])
        Q = np.empty([20,40,30,10,pos_actions])

        total_rewards = []
        total_targets = []

        # --- Code snipped provided for guidance only --- #
        for n in range(epochs):
            # 1) modify parameters
            self.states = []
            self.actions = []
            self.rewards = []

            # 2) create a new drone simulation
            drone = self.init_drone()
            self.nexttarget=drone.get_next_target()
            self.counter = 1

            # 3) run simulation
            for t in range(self.get_max_simulation_steps()):
                drone.set_thrust(self.get_thrusts(drone))
                drone.step_simulation(self.get_time_interval())
            
            states, actions, rewards = self.states, self.actions, self.rewards

            # 4) measure change in quality
            total_new_rewardsblip = np.sum(rewards)
            print(total_new_rewardsblip,self.counter)
            total_rewards.append(total_new_rewardsblip)
            total_targets.append(self.counter)

            G = 0
            # 5) update parameters according to algorithm
            for i in reversed(range(len(actions))):
                k= len(actions)-1
                total_new_rewardsblip
                G = np.average(rewards[(k-i):])
                #OPTION- MAKE THE END OF THE PATH HAVE BIGGER EFFECT replace previous with
                #rewards[k-i]=np.average(rewards[(k-i):])
                #G=rewards[k-i]
                cur_reward = self.value_function[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]


                average_returns[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] += G
                average_returns_count[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] += 1

                Q[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] = average_returns[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] / average_returns_count[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]


                self.value_function[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] = Q[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]

                
                # print("Return " + str(i) + " " + str(average_returns[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]))
                # print("Count " + str(i) + " "  + str(average_returns_count[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]))
                # print("Average " + str(i) + " " + str(Q[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]))
            

        pass

        from datetime import datetime
        import csv

        variablename = {"total reward":total_rewards, "targets hit":total_targets}
        df = pd.DataFrame(variablename)
        now = datetime.now()
        time = now.strftime("Drone-epochs {epochs} %d-%m-%Y_%H-%M-%S")
        df.to_csv(f'{time}.csv')

        plt.figure()
        plt.plot(total_rewards, label="Sampled Mean Return", alpha=1)
        plt.xlabel("Epochs")
        plt.ylabel("Avg Return")
        plt.show()
        
    def graph():
        # plt.figure()
        # plt.plot(range(1+10,len(returns)+1-10), smoothed_returns[10:-10], "-", label="Sampled Mean Return", alpha=1)
        # plt.plot([1, len(returns)+1], [max_return, max_return], '-.', label="Max Return")
        # plt.legend(loc="lower right")
        # plt.xlabel("Epochs")
        # plt.ylabel("Avg Return")
        # plt.show()
        return

    def load(self):
        """Load the parameters of this flight controller from disk.
        """
        try:
            parameter_array = np.load('custom_controller_parameters.npy')
            self.value_function = parameter_array[0]
            print((parameter_array[0]))
        except:
            print("Could not load parameters, sticking with default parameters.")

    def save(self):
        """Save the parameters of this flight controller to disk.
        """
        parameter_array = np.array([self.value_function])
        np.save('custom_controller_parameters.npy', parameter_array)