# Training for agent_m1
# ========================


import pickle
import numpy as np
from typing import List
from codetiming import Timer

import events as e
from .callbacks import state_to_features, features_to_indices
from .callbacks import ACTIONS, model_name


# Constants 
state_count_axis_1  = 15   # number of possible feature states for first / second / third Q axis, currently 15 considering order-invariance
state_count_axis_2  = 3    # OWN POSITION
state_count_axis_3  = 2    # MODI
action_count = len(ACTIONS) # = 6


# Hyperparameters for Q-update
alpha = 0.01   # initially set to 1
gamma = 0.0   # initially set to 1, for now be shortsighted.
mode = "SARSA" # "SARSA" or "Q-Learning"
n = 3 # n-step Q-learning

# Training analysis
Q_file      = lambda x: f"logs/Q_data/Q{x}.npy"
timing_file = f"logs/timing/time_{model_name}.csv"





# Main functions
# --------------

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """


    # Initialize Q
    self.model = self.Q = np.zeros((state_count_axis_1, state_count_axis_2, state_count_axis_3, action_count))   # initial guess for Q, for now just zeros
    
    #self.training_data = []   # [[features, action_index, reward], ...]  
    self.state_indices   = []
    self.sorted_policies = []
    self.rewards         = []
    
    '''
    self.unsorted_policies = [] # debugging purpose
    self.unsorted_features = []
    self.sorted_features = []
    self.tracked_events = []
    '''

    # Logging
    self.logger.debug("str(): Starting training by initializing Q." + '\n' * 2)

    # Time training and log it
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


    self.timer_geo.start()
    
    # Collecting training data: 
    ## new_game_state -> sorted features, 
    ## self_action    -> sorted policy, 
    ## events         -> reward
    '''
    features        = state_to_features(new_game_state)
    sorting_indices = np.argsort(features)   # Moved sorting here to be able to log both sorted and unsorted features.
    sorted_features = features[sorting_indices]
    policy          = ACTIONS.index(self_action)
    sorted_policy   = list(sorting_indices).index(policy)   # find index of self_action, which was actually picked during training
    '''
    reward          = reward_from_events(self, events)   # give auxiliary rewards
    self.rewards.append(reward)
    #self.training_data.append([sorted_features, sorted_policy, reward])

    # self.tracked_events.append(events) # debugging purpose

    # Logging
    #self.logger.debug(f"geo(): Step {new_game_state['step']}")
    self.logger.debug(f'geo(): Encountered game event(s) {", ".join(map(repr, events))}')
    self.logger.debug(f'geo(): Received reward {reward}')

    # Step timing
    geo_time  = self.timer_geo.stop()
    step_time = self.timer_step.stop()
    
    self.step_times.append(step_time)
    self.geo_times.append(geo_time)

    self.timer_step.start()



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


    self.timer_eor.start()

    # Update training data of last round
    ## in the last round there doesn't happen anything except e.'SURVIVED_ROUND'. No actions, no rewards (currently), no need to update anything.
    reward            = reward_from_events(self, events)   # give auxiliary rewards
    self.rewards[-1] += reward
    #self.training_data.append([sorted_features, sorted_policy, reward])

    # self.tracked_events.append(events) # debugging purpose

    # Logging
    #self.logger.debug(f"geo(): Step {new_game_state['step']}")
    self.logger.debug(f'geo(): Encountered game event(s) {", ".join(map(repr, events))}')
    self.logger.debug(f'geo(): Received reward {reward}')
    
    # Updating Q by iterating through every game step
    sum_of_gain_per_Sa = np.zeros_like(self.Q)
    number_of_Sa_steps = np.zeros_like(self.Q)

    
    step_count = last_game_state['step']
    for step in range(step_count):
        # Calculate the state-action pair (S, a)
        state_index_1, state_index_2, state_index_3  = self.state_indices[step]
        sorted_policy = self.sorted_policies[step]
        
        # Update gain for (S, a)
        sum_of_gain_per_Sa[state_index_1, state_index_2, state_index_3, sorted_policy] \
            += Q_update(self, step)
        number_of_Sa_steps[state_index_1, state_index_2, state_index_3, sorted_policy] \
            += 1

    # Average estimated gain per (S, a)
    number_of_Sa_steps[number_of_Sa_steps == 0] = 1   # To fix div-by-zero
    expected_gain_per_Sa = sum_of_gain_per_Sa / number_of_Sa_steps

    # Q-Update
    self.Q = self.Q * (1 - alpha) + alpha * expected_gain_per_Sa
  
    # Save updated Q-function as new model
    self.model = self.Q
    with open(f"model_{model_name}.pt", "wb") as file:
        pickle.dump(self.model, file)

    ''' Debug
    for n in range(10):
        print(self.unsorted_features[n], self.unsorted_policies[n], self.sorted_features[n], self.sorted_policies[n], self.tracked_events[n], self.rewards[n])
    '''

    # Clean up
    #self.training_data = []
    self.state_indices   = []
    self.sorted_policies = []
    self.rewards         = []
    '''
    self.unsorted_policies = [] # debugging purpose
    self.unsorted_features = []
    self.sorted_features = []
    self.tracked_events = []
    '''

    # Training analysis
    ## Logging
    self.logger.debug(f"eor(): Last Step {last_game_state['step']}")
    self.logger.debug(f'eor(): Encountered game event(s) {", ".join(map(repr, events))}')
    #self.logger.debug(f'eor(): Received reward = {reward}')

    ## Save analysis data
    current_round = last_game_state['round']
    with open(Q_file(current_round), "wb") as file:
        np.save(file, self.model)

    ## Time the training and save the data

    eor_time      = self.timer_eor.stop()
    round_time    = self.timer_round.stop()
    self.timer_step.stop()   # This last timer was started after the last step and thus isn't needed.
    
    avg_step_time = np.mean(np.array(self.step_times))
    avg_act_time  = np.mean(np.array(self.act_times))
    avg_geo_time  = np.mean(np.array(self.geo_times))

    ### Appending timing data row-wise to timing csv file
    time_data = np.array([current_round, step_count, 
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
    




# Support functions
# -----------------

def reward_from_events(self, events: List[str]) -> int:
    """
    Here we modify the rewards our agent get so as to en/discourage
    certain behavior.
    """
    
    # Auxiliary Rewards for Task 1
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.INVALID_ACTION: -1,
        e.KILLED_SELF: -1,
        e.OPPONENT_ELIMINATED: 25
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    if not self.train:  self.logger.info(f" Awarded {reward_sum} for events {', '.join(events)}")
    
    return reward_sum

def Q_update(self, t, mode = mode, n = n,  gamma = gamma):
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
    t_plus_n = min(t+n, len(self.rewards)-1)
    v = 0 # initialize just due to a assignment - reference bug otherwise

    # Approximate Q after next n steps
    state_next_n_steps_1, state__next_n_steps_2, state_next_n_steps_3 = self.state_indices[t_plus_n]

    if mode == "Q-Learning":
        v = np.amax(self.Q[state_next_n_steps_1, state__next_n_steps_2, state_next_n_steps_3])
    
    elif mode == "SARSA":
        action_next_n_steps = self.sorted_policies[t_plus_n]
        v = self.Q[state_next_n_steps_1, state__next_n_steps_2, state_next_n_steps_3, action_next_n_steps]
    
    # Compute Reward Sum for next n steps
    reward_sum = 0
    for s in range(t, t_plus_n):
        reward_sum += gamma**(s-t) * self.rewards[s]
    
    return(reward_sum + gamma**n * v)