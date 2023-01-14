import numpy as np
from flight_controller import FlightController
from drone import Drone
from typing import Tuple
from math import log, floor

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

pos_actions=5

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

        # if abs(drone.velocity_x) < 0.1 :
        #     velocity_multiplier = 0
        # else:
        #     velocity_multiplier= clamp(-2*(velocity-0.1)**2+1 ,0.3,1)

        if drone.pitch == 0:
            angle_multiplier = 0.2
        else:
            angle_multiplier= clamp(-0.8*log(abs(drone.pitch)) ,0.1,1)
        
        rotational_multiplier = clamp( -(drone.pitch_velocity/20)**2+1,0.1,1)
        multiplier =  angle_multiplier*angle_multiplier #*rotational_multiplier *velocity_multiplier
        wrongwaypunish= np.sign(dx*drone.pitch)
        wrongwaymulti= clamp(abs(drone.pitch),-1,0.05)
        badboy= 25*floor(abs(drone.pitch))
        #print(drone.pitch)
        reward = -3*(dist) + 0.1*abs(drone.velocity_x) - badboy + 2*self.counter**3 # + wrongwaymulti*wrongwaypunish*5
        #print (multiplier)
        #print(ontarget)
        if (self.nexttarget!=drone.get_next_target()):
            self.nexttarget=drone.get_next_target()
            self.counter+=1
            reward+= 2000+multiplier
        return reward

    def action_selection(self, drone: Drone) -> Tuple[float, float]:
        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y



        if(self.action == 0):        
            # Current
            
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

            thrust_left = np.clip(0.5 + thrust_adj + delta_pitch, 0.0, 1.0)
            thrust_right = np.clip(0.5 + thrust_adj - delta_pitch, 0.0, 1.0)
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
            
        
        return (thrust_left, thrust_right)

    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y

        thrust_adj = np.clip(dy * self.ky, -self.abs_thrust_delta, self.abs_thrust_delta)
        target_pitch = np.clip(dx * self.kx, -self.abs_pitch_delta, self.abs_pitch_delta)
        delta_pitch = target_pitch - drone.pitch
        #print(drone.pitch)
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
        epochs = 300
        # --- Code snipped provided for guidance only --- #
        for n in range(epochs):
            # 1) modify parameters
            
            self.states = []
            self.actions = []
            self.rewards = []


            # 2) create a new drone simulation
            drone = self.init_drone()
            self.nexttarget=drone.get_next_target()
            self.counter=1
            # 3) run simulation
            for t in range(self.get_max_simulation_steps()):
                #print(drone.pitch)
                drone.set_thrust(self.get_thrusts(drone))
                drone.step_simulation(self.get_time_interval())
            #print(self.states)
            states, actions, rewards = self.states, self.actions, self.rewards

            # 4) measure change in quality
            total_new_rewardsblip = np.sum(rewards)
            print(total_new_rewardsblip,self.counter)
            # 5) update parameters according to algorithm

            for i in reversed(range(len(actions))):
                k= len(actions)-1
                total_new_rewardsblip
                rewards[k-i] = (4*np.average(rewards[(k-i):])+total_new_rewardsblip)/5
                # next_reward=total_new_rewards
                # if (k-(i-2)<k):
                #     next_reward = (1/4)*(3*rewards[k-i+1]+rewards[k-i+2])
                # adjusted_new_rewards= (3*(next_reward)+total_new_rewards)/4
                cur_reward = self.value_function[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]]
                # #print(states[i])
                # new_reward = (5*cur_reward + (adjusted_new_rewards/len(rewards[k-i:])))/ 6
                if (cur_reward>rewards[k-i]):
                    new_reward= (19*cur_reward + rewards[k-i])/20
                else:
                    new_reward= (1*cur_reward + rewards[k-i])/5
                self.value_function[states[k-i][0]][states[k-i][1]][states[k-i][2]][states[k-i][3]][actions[k-i]] = new_reward

            # for i in range(len(actions)):
            #     total_new_rewards= np.sum(rewards[i:])/len(rewards[i:])
            #     next_reward=total_new_rewards
            #     if (i+2<len(actions)):
            #         next_reward = (1/4)*(3*rewards[i+1]+rewards[i+2])
            #     adjusted_new_rewards= ((next_reward)+3*total_new_rewards)/4
            #     cur_reward = self.value_function[states[i][0]][states[i][1]][states[i][2]][states[i][3]][actions[i]]
            #     #print(states[i])
            #     new_reward = (5*cur_reward + (adjusted_new_rewards/len(rewards[i:])))/ 6
                
            #     self.value_function[states[i][0]][states[i][1]][states[i][2]][states[i][3]][actions[i]] = new_reward
        pass

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