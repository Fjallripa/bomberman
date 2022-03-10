import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features, find_state
from .callbacks import ACTIONS, model_file


# constants 
state_count  = 81   # number of possible feature states, currently 81
action_count = len(ACTIONS)   # 6

# Hyperparameters for Q-update
alpha = 1   # initially set to 1
gamma = 1   # initially set to 1



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """


    # Initialize Q
    self.logger.debug("Starting training by initializing Q.")
    self.Q = np.zeros((state_count, action_count))   # initial guess for Q, for now just zeros
    
    self.training_data = []   # [[features, action_index, reward], ...]
    self.state_occurances = np.zeros_like(self.Q)   # a counter for how often the individual game_states happened during training



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

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    features = state_to_features(old_game_state)
    action   = np.find(ACTIONS, self_action)   # find index of self_action
    reward   = new_game_state['self'][1] - old_game_state['self'][1]   # just game reward for now, reward_from_events() better place for training reward calculations

    self.training_data.append([features, action, reward])



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

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # no update to self.training_data here (game_events_occured() also gets called after the last action, doesn't it?)
    
    game_length = len(self.training_data)

    features_old, action_old, reward_new    = self.training_data[0]
    state_index_old                         = find_state(features_old)
    self.state_occurances[state_index_new] += 1
    
    for step in range(1, game_length):
        # Preparation
        features_new, action_new, reward_next   = self.training_data[step]
        state_index_new                         = find_state(features_new)
        self.state_occurances[state_index_new] += 1

        Q_state_old  = self.Q[state_index_old][action_old]
        Q_state_new  = self.Q[state_index_new][action_new]
        V_state_new  = np.max(Q_state_new)   # implemented Q-learning instead of SARSA
        N_Sa         = self.state_occurances[state_index_new]
        
        # Q-Update
        self.Q[state_index_old][action_old] = Q_state_old + alpha * (reward_new + gamma / N_Sa * V_state_new - Q_state_old)

        # new state becomes old state
        state_index_old = state_index_new
        action_old      = action_new
        reward_new      = reward_next
        

    # Save updated Q-function as new model
    self.model = self.Q
    with open(model_file, "wb") as file:
        pickle.dump(self.model, file)




def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    
    return reward_sum
