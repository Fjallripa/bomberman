# Training for agent_h2
# =====================


import pickle
import numpy as np
from typing import List
#from codetiming import Timer

import events as e
from .callbacks import AGENT_NAME, MODEL_NAME, model_file
from .callbacks import ACTIONS
from .callbacks import ALPHA, GAMMA, MODE, N





# Global training constants
# -------------------------

# Q-model constants 
state_count_axis_1 = 15   # number of possible feature states for first / second / third Q axis, currently 15 considering order-invariance
state_count_axis_2 = 3    # OWN POSITION
state_count_axis_3 = 3    # MODI
action_count       = len(ACTIONS)   # = 6

# Training analysis files
Q_file  = lambda x: f"logs/Q_data/Q{x}.npy"
Sa_file = "logs/state_action_counter.npy"
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
    self.model = self.Q = self.Sa_counter = \
        np.zeros((state_count_axis_1, state_count_axis_2, state_count_axis_3, action_count))   # initial guess for Q, for now just zeros

    # Initialize training data lists
    self.state_indices   = []
    self.sorted_policies = []
    self.rewards         = []
    
    # Logging
    self.logger.debug("str(): Starting training by initializing Q." + '\n' * 2)

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
    reward          = reward_from_events(self, events)   # give auxiliary rewards
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
    round_length = len(self.state_indices)
    if round_length == 400:
        self.rewards[-1] += reward
    else:
        self.rewards.append(reward)
    
    
    ## Logging
    self.logger.debug(f"eor(): Last Step {round_length}")
    self.logger.debug(f'eor(): Encountered game event(s) {", ".join(map(repr, events))}')
    self.logger.debug(f'eor(): Received reward = {reward}')


    # Updating Q by iterating through every game step
    sum_of_gain_per_Sa = np.zeros_like(self.Q)
    number_of_Sa_steps = np.zeros_like(self.Q)


    for step in range(round_length):
        # Calculate the state-action pair (S, a)
        state_indices = self.state_indices[step]
        sorted_policy = self.sorted_policies[step]
        
        # Update gain for (S, a)
        sum_of_gain_per_Sa[state_indices][sorted_policy] += Q_update(self, step)
        number_of_Sa_steps[state_indices][sorted_policy] += 1

    self.Sa_counter += number_of_Sa_steps  # Collects total number of encounters with each (S, a) during training.


    # Q-Update
    occured_Sa         = number_of_Sa_steps != 0 # True if Sa occured in last round, else False 
    self.Q[occured_Sa] = self.Q[occured_Sa] * (1 - ALPHA) + ALPHA * sum_of_gain_per_Sa[occured_Sa] / number_of_Sa_steps[occured_Sa]
  
  
    # Save updated Q-function as new model
    self.model = self.Q
    with open(model_file, "wb") as file:
        pickle.dump(self.model, file)

    
    # Clean up
    self.state_indices   = []
    self.sorted_policies = []
    self.rewards         = []
    

    # Training analysis

    ## Save analysis data
    current_round = last_game_state['round']
    with open(Q_file(current_round), "wb") as file:
        np.save(file, self.model)
    np.save(Sa_file, self.Sa_counter)

    

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



def Q_update(self, t, mode = MODE, n = N,  gamma = GAMMA):
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
    
    
    t_plus_n = min(t+n, len(self.state_indices)-1)
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
