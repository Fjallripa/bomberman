# Training for agent_swq12
# ========================


import pickle
import numpy as np
from typing import List
from codetiming import Timer

import events as e
from .callbacks import state_to_features, features_to_indices
from .callbacks import ACTIONS, model_name


# Constants 
state_count  = 15   # number of possible feature states, currently 15 considering order-invariance
action_count = 4 # was previously & should be in general = len(ACTIONS) = 6; changed for task 1; shouldn't be changed without changing feature design & act()


# Hyperparameters for Q-update
alpha = 1/5   # initially set to 1
gamma = 0.5   # initially set to 1

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
    self.Q_old = np.zeros((state_count, action_count))   # initial guess for Q, for now just zeros
    
    self.training_data = []   # [[features, action_index, reward], ...]  

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
    :param old_game_state: The state before the new_game_state.
    :param self_action: The action that you took.
    :param new_game_state: The state that was passed to the last call of `act`.
    :param events: The events that occurred when going from `new_game_state` to next game state.
    """


    self.timer_geo.start()
    
    # Collecting training data: 
    ## new_game_state -> sorted features, 
    ## self_action    -> sorted policy, 
    ## events         -> reward
    sorted_features, sorting_indices  = state_to_features(new_game_state)
    policy          = ACTIONS.index(self_action)
    sorted_policy   = list(sorting_indices).index(policy)  # find index of self_action, which was actually picked during training
    reward          = reward_from_events(self, events)   # give auxiliary rewards
    
    self.training_data.append([sorted_features, sorted_policy, reward])

    # Logging
    self.logger.debug(f"geo(): Step {new_game_state['step']}")
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
    
    # Updating Q by iterating through whole training data
    self.Q_new = np.zeros_like(self.Q_old) # initialize
    counter = np.zeros_like(self.Q_old)

    for t, training_data in enumerate(self.training_data):
        this_state = features_to_indices(training_data[0])
        this_action = training_data[1]
        self.Q_new[this_state, this_action] += self.Q_old[this_state, this_action] * (1 - alpha) \
            + alpha * Q_update(self, t, n = 3, mode = "SARSA") # change learning mode (n-step, Q-learning vs. SARSA) here
        counter[this_state, this_action] += 1

    self.Q_old = self.Q_new / np.where(counter != 0, counter, 1) # fix singularity bug

    # Save & clean up
    self.model = self.Q_old
    with open(f"model_{model_name}.pt", "wb") as file: 
        pickle.dump(self.model, file)

    self.training_data = []

    """

    # Updating Q by iterating through every game step
    #self.Q_new = self.Q_old
    
    ## step 0
    features_old, action_old, reward_old \
                    = self.training_data[0]
    state_index_old = features_to_indices(features_old)
    
    ## step 1..end
    step_count = last_game_state['step']
    for step in range(1, step_count):
        # Preparation
        features_new, action_new, reward_new \
                        = self.training_data[step]
        state_index_new = features_to_indices(features_new)
        
        Q_state_old  = self.Q_old[state_index_old][action_old]
        V_state_new  = np.amax(self.Q_old[state_index_new])   # implemented Q-learning instead of SARSA
        
        # Q-Update
        #self.Q_new...
        self.Q_old[state_index_old][action_old] = Q_state_old + alpha * (reward_old + gamma * V_state_new - Q_state_old)  # also try alpha / N_Sa

        # New state becomes old state
        state_index_old = state_index_new
        action_old      = action_new
        reward_old      = reward_new
    

    # Save updated Q-function as new model
    #self.Q_old = self.Q_new
    self.model = self.Q_old
    with open(f"model_{model_name}.pt", "wb") as file:
        pickle.dump(self.model, file)

    # Clean up
    self.training_data = []

    """

    # Training analysis
    ## Logging
    self.logger.debug(f"eor(): Last Step {last_game_state['step']}")
    self.logger.debug(f'eor(): Encountered game event(s) {", ".join(map(repr, events))}')
    #self.logger.debug(f'eor(): Received reward = {reward}')

    ## Save analysis data
    current_round = last_game_state['round']
    with open(Q_file(current_round), "wb") as file:
        np.save(file, self.Q_old)

    ## Time the training and save the data

    eor_time      = self.timer_eor.stop()
    round_time    = self.timer_round.stop()
    self.timer_step.stop()   # This last timer was started after the last step and thus isn't needed.
    
    avg_step_time = np.mean(np.array(self.step_times))
    avg_act_time  = np.mean(np.array(self.act_times))
    avg_geo_time  = np.mean(np.array(self.geo_times))

    ### Appending timing data row-wise to timing csv file
    step_count = last_game_state['step'] # added for logging
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
        e.INVALID_ACTION: -1
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    if not self.train:  self.logger.info(f" Awarded {reward_sum} for events {', '.join(events)}")
    
    return reward_sum

def Q_update(self, t, mode = "Q-Learning", n = 1,  gamma = gamma):
    """
    Computes the new value during Q-learning.

    Input:
    ======
    self: 
    n: for n-step Q-learning
    gamma: hyperparameter
    mode: "SARSA" or "Q-Learning"

    Output:
    =======
    the value to update Q, a scalar
    """
    t_plus_n = min(t+n, len(self.training_data)-1)

    # Approximate Q after next n steps
    state_next_n_steps = features_to_indices(self.training_data[t_plus_n][0])

    if mode == "Q-Learning":
        v = np.amax(self.Q_old[state_next_n_steps])
    
    elif mode == "SARSA":
        action_next_n_steps = self.training_data[t_plus_n][1]
        v = self.Q_old[state_next_n_steps, action_next_n_steps]
    
    # Compute Reward Sum for next n steps
    reward_sum = 0
    for s in range(t, t_plus_n):
        reward_sum += gamma**(s-t) * self.training_data[s][2]
    
    return(reward_sum + gamma**n * v)






