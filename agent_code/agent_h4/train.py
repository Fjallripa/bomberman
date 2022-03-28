# Training for agent_h4
# =====================


import os
from telnetlib import DO
from turtle import update
import numpy as np
from typing import List
#from codetiming import Timer

from .callbacks import MODEL_FILE, SA_COUNTER_FILE
from .callbacks import ALPHA, GAMMA, MODE, N, REWARDS, DOUBLE_Q_LEARNING
from .callbacks import START_TRAINING_WITH, AGENT_NAME, Q_SAVE_INTERVAL, Q_UPDATE_INTERVAL
from .callbacks import random_argmax





# Global training constants
# -------------------------

# Q-model constants 
state_count_axis_1 = 15   # number of possible feature states for first / second Q axis, currently 15 considering order-invariance
state_count_axis_2 = 3    # OWN POSITION
action_count       = 6    # len(ACTIONS)

# Training analysis files
Q_file  = lambda x: f"logs/Q_data/Q{x}.npy"
#timing_file = f"logs/timing/time_{AGENT_NAME}_{MODEL_NAME}.csv"





# Main functions
# --------------

def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """


    # Initialize Q and state-action-counter
    if START_TRAINING_WITH == "RESET":
        self.logger.info(f"Training from scratch")
        self.logger.info(f"Initializing Q and Sa_counter with zeros.")
        
        self.model      = np.zeros((state_count_axis_1, state_count_axis_2, action_count))
        self.Q_A        = np.zeros_like(self.model)
        self.Q_B        = np.zeros_like(self.model)
        self.Sa_counter = np.zeros_like(self.model, dtype = int)
    else:
        Q_INITIAL          = f"models/model_{AGENT_NAME}_{START_TRAINING_WITH}.npy"
        SA_COUNTER_INITIAL = f"models/sa_counter_{AGENT_NAME}_{START_TRAINING_WITH}.npy"
        
        if not os.path.isfile(Q_INITIAL):
            print(f"\nERROR: the inital Q-file {Q_INITIAL} couldn't be found!\n")
        if not os.path.isfile(SA_COUNTER_INITIAL):
            print(f"\nERROR: the inital Sa-counter-file {SA_COUNTER_INITIAL} couldn't be found!\n")

        self.logger.info(f"Loading inital Q from {Q_INITIAL}.")
        self.logger.info(f"Loading inital Sa_counter from {SA_COUNTER_INITIAL}.")
        
        self.model      = np.load(Q_INITIAL)
        self.Q_A        = np.load(Q_INITIAL)
        self.Q_B        = np.load(Q_INITIAL)
        self.Sa_counter = np.load(SA_COUNTER_INITIAL)
        

    
    # Initialize training data lists
    self.state_indices   = []
    self.sorted_policies = []
    self.rewards         = []
    self.round_lengths   = np.zeros(Q_UPDATE_INTERVAL, dtype = int)
    

    # Time training and log it
    '''
    timing_header = "\t".join(['round', 'step_count', 
                               'round_time', 'avg_step_time', 
                               'avg_act_time', 'avg_geo_time',
                               'eor_time'])
    with open(timing_file, 'w') as file:
        np.savetxt(file, np.array([]), header = timing_header, comments = "")

    self.step_times = []
    self.act_times  = []
    self.geo_times  = []
   
    self.timer_round = Timer(logger = None)
    self.timer_step  = Timer(logger = None)
    self.timer_act   = Timer(logger = None)
    self.timer_geo   = Timer(logger = None)
    self.timer_eor   = Timer(logger = None)
    
    self.timer_round.start()
    self.timer_step.start()
    '''    



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """


    #self.timer_geo.start()
    
    # Calculating rewards
    reward = reward_from_events(self, events)   # give auxiliary rewards
    self.rewards.append(reward)
    
    # Logging
    self.logger.debug(f'geo(): Encountered game event(s) {", ".join(map(repr, events))}')
    self.logger.debug(f'geo(): Received reward {reward}')

    # Step timing
    '''
    geo_time  = self.timer_geo.stop()
    step_time = self.timer_step.stop()
    
    self.step_times.append(step_time)
    self.geo_times.append(geo_time)

    self.timer_step.start()
    '''



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """


    #self.timer_eor.start()

    # Update training data of last round
    reward       = reward_from_events(self, events)   # give auxiliary rewards
    round_length = len(self.state_indices) - np.sum(self.round_lengths)
    if round_length == 400:
        self.rewards[-1] += reward
    else:
        self.rewards.append(reward)

    ## For Q_UPDATE_INTERVAL != 0, save the current round length
    current_round = last_game_state['round']
    round_index   = (current_round - 1) % Q_UPDATE_INTERVAL
    self.round_lengths[round_index] \
                  = round_length
    
    
    ## Logging
    self.logger.debug(f"eor(): Last Step {round_length}")
    self.logger.debug(f'eor(): Encountered game event(s) {", ".join(map(repr, events))}')
    self.logger.debug(f'eor(): Received reward = {reward}')


    # Alernatingly updating Q_A and Q_B by collecting rewards from all encountered states and estimating their gain.
    if current_round % Q_UPDATE_INTERVAL == 0:
        current_update = (current_round - 1) // Q_UPDATE_INTERVAL + 1
        update_length  = len(self.state_indices)
        cumulative_round_lengths = np.cumsum(self.round_lengths)
        
        sum_of_gain_per_Sa = np.zeros_like(self.model)
        number_of_Sa_steps = np.zeros_like(self.model, dtype = int)

        for step_index in range(update_length):
            # Index of next round
            index_next_round      = np.sum(step_index >= cumulative_round_lengths)
            step_count_next_round = self.round_lengths[index_next_round]

            # Calculate the state-action pair (S, a)
            state_indices = self.state_indices[step_index]
            sorted_policy = self.sorted_policies[step_index]
            
            # Update gain for (S, a)
            sum_of_gain_per_Sa[state_indices][sorted_policy] += Q_update(self, step_index, step_count_next_round, current_update)
            number_of_Sa_steps[state_indices][sorted_policy] += 1

        self.Sa_counter += number_of_Sa_steps   # Collects total number of encounters with each (S, a) during training.


        # Q-Update
        occured_Sa    = number_of_Sa_steps != 0   # True if Sa occured in last round, else False 
        expected_gain = sum_of_gain_per_Sa[occured_Sa] / number_of_Sa_steps[occured_Sa]
        
        if current_update % 2 == 1:  self.Q_A[occured_Sa] = (1 - ALPHA) * self.Q_A[occured_Sa] + ALPHA * expected_gain
        else:                        self.Q_B[occured_Sa] = (1 - ALPHA) * self.Q_B[occured_Sa] + ALPHA * expected_gain
    
    
        # Save updated Q-function as new model
        self.model = self.Q_A  if (current_update % 2 == 1)  else self.Q_B
        if not DOUBLE_Q_LEARNING: self.Q_A = self.Q_B = self.model  # Hopefully, syncronizing Q_A and Q_B undoes Double-Q-Learning
        np.save(MODEL_FILE, self.model)
        np.save(SA_COUNTER_FILE, self.Sa_counter)

    
        # Reset training data
        self.state_indices   = []
        self.sorted_policies = []
        self.rewards         = []
        self.round_lengths   = np.zeros(Q_UPDATE_INTERVAL, dtype = int)
    

        # Training analysis

        ## Save analysis data
        if current_round % Q_SAVE_INTERVAL == 0:
            np.save(Q_file(current_round), self.model) 

    ## Time the training and save the data
    '''
    eor_time      = self.timer_eor.stop()
    round_time    = self.timer_round.stop()
    self.timer_step.stop()   # This last timer was started after the last step and thus isn't needed.
    
    avg_step_time = np.mean(np.array(self.step_times))
    avg_act_time  = np.mean(np.array(self.act_times))
    avg_geo_time  = np.mean(np.array(self.geo_times))

    ### Appending timing data row-wise to timing csv file
    time_data = np.array([current_round, round_length, 
                          round_time, avg_step_time, 
                          avg_act_time, avg_geo_time,
                          eor_time], ndmin = 2)   # 1xn matrix to become a row
    with open(timing_file, 'a') as file:
        np.savetxt(file, time_data, delimiter = '\t')

    ### Clean up
    self.step_times = []
    self.act_times  = []
    self.geo_times  = []

    ### Starting timer again for the next round.
    self.timer_round.start()
    self.timer_step.start()
    '''





# Support functions
# -----------------

def reward_from_events(self, events: List[str]) -> int:
    """
    Here we modify the rewards our agent get so as to en/discourage
    certain behavior.
    """
    

    game_rewards = REWARDS
    reward_sum   = 0
    
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    if not self.train:  self.logger.info(f" Awarded {reward_sum} for events {', '.join(events)}")
    
    return reward_sum



def Q_update(self, t, next_round, update_number, mode = MODE, n = N,  gamma = GAMMA):
    """
    Computes the new value during Q-learning.

    Input:
    ======
    self: 
    t: step of action
    n: for n-step Q-learning
    gamma: hyperparameter
    mode: "SARSA" or "Q-Learning"

    Output:
    =======
    the value to update Q, a scalar
    """
    
    
    # Initializations
    t_plus_n   = min(t+n, next_round-1)
    v          = 0
    reward_sum = 0

    # Approximate Q after next n steps
    state_indices = self.state_indices[t_plus_n]

    if mode == "Q-Learning":
        if update_number % 2 == 1:
            best_action_A = random_argmax(self.Q_A[state_indices])
            v             = self.Q_B[state_indices][best_action_A]
        else:
            best_action_B = random_argmax(self.Q_B[state_indices])
            v             = self.Q_A[state_indices][best_action_B]
    
    elif mode == "SARSA":
        chosen_action = self.sorted_policies[t_plus_n]
        if update_number % 2 == 1:  v = self.Q_B[state_indices][chosen_action]
        else:                       v = self.Q_A[state_indices][chosen_action]
    
    # Compute Reward Sum for next n steps
    for s in range(t, t_plus_n):
        reward_sum += gamma**(s-t) * self.rewards[s]
    
    return(reward_sum + gamma**n * v)
