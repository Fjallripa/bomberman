# Callbacks for agent_h4
# ======================


import os
import json
import random
import numpy as np

import events as e
from settings import SCENARIOS





# Global Constants
# ----------------

# Changeable Parameters

## Training parameters - CHANGE FOR EVERY TRAINING
AGENT_NAME          = "h4"
MODEL_NAME          = "coin-miner4"
SCENARIO            = "loot-box"
OTHER_AGENTS        = []
TRAINING_ROUNDS     = 1500
START_TRAINING_WITH = "coin-collector1"   # "RESET" or "<model_name>"

## Hyperparameters for epsilon-annealing - CHANGE IF YOU WANT
EPSILON_MODE             = "experience"
if EPSILON_MODE == "experience":
    EPSILON_AT_START     = 1
    EPSILON_THRESHOLD    = 0.1
    EPSILON_AT_INFINITY  = 0.001
    THRESHOLD_EXPERIENCE = 2000
if EPSILON_MODE == "rounds":
    EPSILON_AT_ROUND_ZERO = 1
    EPSILON_THRESHOLD     = 0.1
    EPSILON_AT_INFINITY   = 0
    THRESHOLD_FRACTION    = 0.33
if EPSILON_MODE == "old":
    EPSILON_AT_ROUND_ZERO = 0.3
    EPSILON_AT_ROUND_LAST = 0.001

## Hyperparameters for Q-update - CHANGE IF YOU WANT
ALPHA = 0.1
GAMMA = 0.9
MODE  = "Q-Learning"   # "SARSA" or "Q-Learning"
N     = 5         # N-step Q-learning

## Hyperparameters for agent behavior - CHANGE IF YOU WANT
FOE_TRIGGER_DISTANCE = 5
STRIKING_DISTANCE    = 3

## Rewards
REWARDS = {
    e.COIN_COLLECTED: 5,
    e.INVALID_ACTION: -1,
    e.KILLED_OPPONENT: 100,
    e.GOT_KILLED: -1,    
}



# Fixed and derived constants - DON'T CHANGE UNLESS JUSTIFIED

## Game constants
ACTIONS              = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS           = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])   # UP, RIGHT, DOWN, LEFT
DEFAULT_DISTANCE     = 1000
BOMB_COOLDOWN_TIME   = 7
COLS = ROWS          = 17
BLAST                = np.array([-3, -2, -1, 1, 2, 3])

## Calculate constant BOMB_MASK one time
BOMB_MASK = np.full((COLS, ROWS, COLS, ROWS), False)

x_inside = lambda x: x > 0 and x < COLS-1
y_inside = lambda y: y > 0 and y < ROWS-1

for x in range(1, COLS-1):
        for y in range(1, ROWS-1):
            if (x % 2 == 1 or y % 2 == 1):
                explosion_spots = [(x, y)]
                if x % 2 == 1:
                    explosion_spots += [(x, y + b) for b in BLAST  if y_inside(y + b)]
                if y % 2 == 1:
                    explosion_spots += [(x + b, y) for b in BLAST  if x_inside(x + b)]
                
                explosion_spots = tuple(np.array(explosion_spots).T)
                BOMB_MASK[(x, y)][explosion_spots] \
                                = True

## Derive model file name
MODEL_FILE      = f"models/model_{AGENT_NAME}_{MODEL_NAME}.npy"
SA_COUNTER_FILE = f"models/sa_counter_{AGENT_NAME}_{MODEL_NAME}.npy"

## Derive scenario settings
COINS = SCENARIOS[SCENARIO]['COIN_COUNT']

## Derive constants for epsilon annealing
if EPSILON_MODE == "experience":
    A = EPSILON_AT_START - EPSILON_AT_INFINITY
    L = 1 / THRESHOLD_EXPERIENCE * np.log(A / (EPSILON_THRESHOLD - EPSILON_AT_INFINITY))
if EPSILON_MODE == "rounds":
    A               = EPSILON_AT_ROUND_ZERO - EPSILON_AT_INFINITY
    ROUND_THRESHOLD = int(TRAINING_ROUNDS * THRESHOLD_FRACTION)
    L               = 1 / ROUND_THRESHOLD * np.log(A / (EPSILON_THRESHOLD - EPSILON_AT_INFINITY))
if EPSILON_MODE == "old":
    EPSILON_AT_ROUND_ONE  = np.power(EPSILON_AT_ROUND_LAST / EPSILON_AT_ROUND_ZERO, 
                                    1 / TRAINING_ROUNDS)  # n-th root of epsilon_at_last_round



# Main functions
# --------------

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train:
       # Save traing related parameters in json file
        params_file = 'logs/params.json'
        params             = {}
        params['training'] = {}
        params['epsilon']  = {}
        params['Q-update'] = {}
        params['agent']    = {}

        params['training']['AGENT_NAME']          = AGENT_NAME
        params['training']['MODEL_NAME']          = MODEL_NAME
        params['training']['SCENARIO']            = SCENARIO
        params['training']['OTHER_AGENTS']        = OTHER_AGENTS
        params['training']['TRAINING_ROUNDS']     = TRAINING_ROUNDS
        params['training']['START_TRAINING_WITH'] = START_TRAINING_WITH
        params['epsilon']['EPSILON_MODE']              = EPSILON_MODE
        if EPSILON_MODE == "experience":
            params['epsilon']['EPSILON_AT_START']      = EPSILON_AT_START
            params['epsilon']['EPSILON_THRESHOLD']     = EPSILON_THRESHOLD
            params['epsilon']['EPSILON_AT_INFINITY']   = EPSILON_AT_INFINITY
            params['epsilon']['THRESHOLD_EXPERIENCE']  = THRESHOLD_EXPERIENCE
        if EPSILON_MODE == "rounds":
            params['epsilon']['EPSILON_AT_ROUND_ZERO'] = EPSILON_AT_ROUND_ZERO
            params['epsilon']['EPSILON_THRESHOLD']     = EPSILON_THRESHOLD
            params['epsilon']['EPSILON_AT_INFINITY']   = EPSILON_AT_INFINITY
            params['epsilon']['THRESHOLD_FRACTION']    = THRESHOLD_FRACTION
        if EPSILON_MODE == "old":
            params['epsilon']['EPSILON_AT_ROUND_ZERO'] = EPSILON_AT_ROUND_ZERO
            params['epsilon']['EPSILON_AT_ROUND_LAST'] = EPSILON_AT_ROUND_LAST
        params['Q-update']['ALPHA'] = ALPHA
        params['Q-update']['GAMMA'] = GAMMA
        params['Q-update']['MODE']  = MODE
        params['Q-update']['N']     = N
        params['agent']['FOE_TRIGGER_DISTANCE'] = FOE_TRIGGER_DISTANCE
        params['agent']['STRIKING_DISTANCE']    = STRIKING_DISTANCE
        params['agent']['COINS']                = COINS   
        params['rewards'] = REWARDS         
        
        with open(params_file, 'w') as file:
            json.dump(params, file, indent = 4)

        # Rest of setup done in setup_training()
        

    elif not os.path.isfile(MODEL_FILE):
        print(f"\nError: the model file {MODEL_FILE} couldn't be found!\n")

    else:
        self.logger.info(f"Loading model {MODEL_FILE} from saved state.")
        self.model = np.load(MODEL_FILE)



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    Currently: works only for movement actions, i.e. Actions[:4]
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    

    round = game_state['round']
    self.logger.debug(f"act(): Round {round}, Step {game_state['step']}:")
    
    #if self.train:  self.timer_act.start()

    # Calculate Q-state (`state_indices`) from game_state
    features                  = state_to_features(self, game_state)
    direction_features        = features[:4]
    sorting_indices           = np.argsort(direction_features)   # Moved sorting here to be able to log both sorted and unsorted features.
    sorted_direction_features = direction_features[sorting_indices]
    state_indices             = features_to_indices(sorted_direction_features), features[4], features[5]

    # Determine action to be taken
    if self.train:
        # Calculate probability of random action (`eps`)
        if EPSILON_MODE == "experience":
            seen_this_n_times_before = np.sum(self.Sa_counter[state_indices])
            eps                      = epsilon(seen_this_n_times_before)
        elif EPSILON_MODE == "rounds":
            eps                      = epsilon(round)
        elif EPSILON_MODE == "old":
            eps                      = epsilon_old(round)

        # Choose policy action with probability 1 - eps
        sorted_policy, label \
               = epsilon_greedy(random_argmax(self.model[state_indices]), eps)
        policy = np.append(sorting_indices, np.array([4,5]))[sorted_policy]
        action = ACTIONS[policy]
        self.state_indices.append(state_indices)
        self.sorted_policies.append(sorted_policy)

    else:
        # Choose policy action
        sorted_policy = random_argmax(self.model[state_indices])
        policy        = np.append(sorting_indices, np.array([4,5]))[sorted_policy]
        action        = ACTIONS[policy]
        label         = "policy"

    # Logging
    current_state  = f"act(): Game State: Position {game_state['self'][3]}, Features {features}"
    current_state += f", epsilon {eps:.3f}"  if self.train  else ""
    self.logger.debug(current_state)
    self.logger.debug(f"act(): Symmetry: Sorted features {np.append(sorted_direction_features, features[4:])}, Q-indeces {state_indices}, Sorted policy {sorted_policy}")
    self.logger.debug(f"act(): Performed {label} action {action}")
    
    # Timing this function
    '''
    if self.train: 
        act_time = self.timer_act.stop()
        self.act_times.append(act_time)
    '''
    
    return action 





# Support functions
# -----------------

def epsilon (occurances):
    return A * np.exp(- L * occurances) + EPSILON_AT_INFINITY


def epsilon_old(round):
    return EPSILON_AT_ROUND_ZERO * np.power(EPSILON_AT_ROUND_ONE, round)



def state_to_features (self, game_state):
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e. a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    
    
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None


    # 0. Collect relevant game_state info
    own_position      = game_state['self'][3]
    crate_map         = game_state['field']
    can_place_bomb    = game_state['self'][2]
    collectable_coins = game_state['coins']
    bombs             = game_state['bombs']
    explosion_map     = game_state['explosion_map']
    foes              = game_state['others']

    neighbors = own_position + DIRECTIONS

    ## Define variables for updates & reset for each round
    if game_state['step'] == 1:
        self.dumb_bombing_map           = np.zeros((COLS, ROWS)) 
        self.cannot_bomb_ticker         = 0
        self.previous_collectable_coins = []
        self.already_collected_coins    = 0
        
        self.logger.debug(f'stf(): Updated dumb_bombing_map and previous_collectable_coins.')

    ## Count already collected coins
    for coin in self.previous_collectable_coins: 
        if coin not in collectable_coins:
            self.already_collected_coins += 1
    self.logger.debug(f'stf(): # coins collected = {self.already_collected_coins}')
    

    # 1. Calculate proximity map
    distance_map, reachability_map, direction_map = proximity_map(own_position, crate_map)


    # 2. Check for danger and lethal danger
    """
    going_is_dumb   : 
        Intuitive definition : going in that direction is suicidal
        Effect if True       : sets f1-f4 to 0
    waiting_is_dumb : 
        Intuitive definition : staying on own_position is suicidal
        Effect if True       : sets f5 to 0
    bombing_is_dumb : 
        Intuitive definition : placing a bomb on own_position is suicidal?
        Effect if True       : prevents f5 from being 2 via best_crate_bombing_spots() (only mode 1)
    """
    
    waiting_is_dumb = False
    bombing_is_dumb = False
    
    # Don't go where it's invalid or suicidal
    going_is_dumb   = np.array([( (not reachability_map[(x,y)]) or explosion_map[(x,y)] ) for [x,y] in neighbors])
    
    # Don't place a bomb if you're not able to
    if not can_place_bomb: 
        bombing_is_dumb = True 
        self.cannot_bomb_ticker = (self.cannot_bomb_ticker % 6) + 1
        if self.cannot_bomb_ticker == 1:
            self.dumb_bombing_map = np.zeros((COLS, ROWS)) # forget all memorized dumb bombing spots 
            self.logger.debug(f'stf(): Updated dumb_bombing_map.')
    
    # Escape bombs that are about to explode
    for (bomb_position, bomb_timer) in bombs:
        steps_until_explosion    = bomb_timer + 1
        no_future_explosion_mask = np.logical_not(BOMB_MASK[bomb_position])

        if not waiting_is_dumb:
            rescue_distances         = distance_map[reachability_map & no_future_explosion_mask]   # improve by including explosions
            minimal_rescue_distance  = DEFAULT_DISTANCE if (rescue_distances.size == 0) else np.amin(rescue_distances)
            if steps_until_explosion <= minimal_rescue_distance:
                waiting_is_dumb = True
                bombing_is_dumb = True

        safe_directions = np.any(direction_map[reachability_map & no_future_explosion_mask & (distance_map <= steps_until_explosion)], axis = 0)
        going_is_dumb[np.logical_not(safe_directions)] = True

    # Don't place a bomb you can't escape from
    if not bombing_is_dumb:
        no_future_explosion_mask = np.logical_not(BOMB_MASK[own_position])
        rescue_distances         = distance_map[reachability_map & no_future_explosion_mask] # improve by including explosions
        minimal_rescue_distance  = DEFAULT_DISTANCE if (rescue_distances.size == 0) else np.amin(rescue_distances) 
        if minimal_rescue_distance > 4:
            bombing_is_dumb = True
            self.dumb_bombing_map[own_position] = bombing_is_dumb 
    
    # 3. Check game mode
    reachable_coins = select_reachable(collectable_coins, reachability_map)
    if len(foes) > 0:
        foe_positions       = np.array([foe[3] for foe in foes])
        foe_positions_tuple = tuple(foe_positions.T)
        min_foe_distance    = np.amin(distance_map[foe_positions_tuple])
    else:
        foe_positions       = np.array([])
        min_foe_distance    = DEFAULT_DISTANCE    
    
    if min_foe_distance <= FOE_TRIGGER_DISTANCE or self.already_collected_coins == COINS:
        mode = 2   # Hunter mode
    elif len(reachable_coins) > 0:
        mode = 0   # Collector mode
    else:
        mode = 1   # Miner mode

 
    # 4. Compute goal direction
    if mode == 0:
        coins_around_me = exclude_from(collectable_coins, own_position)  # Don't have a coin goal on f5
        coins_i_reach   = select_reachable(coins_around_me, reachability_map)
        best_coins      = select_nearest(coins_i_reach, distance_map)
        goals           = make_goals(best_coins, direction_map, own_position)

    if mode == 1:
        crates_destroyed     = crate_destruction_map(crate_map, bombs)
        sensible_bombing_map = sensible_bombing_spots(reachability_map, self.dumb_bombing_map)
        best_crate_spots     = best_crate_bombing_spots(distance_map, crates_destroyed, sensible_bombing_map)
        goals                = make_goals(best_crate_spots, direction_map, own_position)

    if mode == 2:
        closest_foe = select_nearest(foe_positions, distance_map)
        goals       = make_goals(closest_foe, direction_map, own_position)
        if min_foe_distance <= STRIKING_DISTANCE and not bombing_is_dumb:
            goals[4] = True  

    # 5. Assemble feature array
    features = np.full(5, 1)
    
    # Directions (f1 - f4)
    for i in range(4):
        neighbor = tuple(np.array(own_position) + DIRECTIONS[i])
        if going_is_dumb[i]:
            features[i] = 0
        elif goals[i]:
            features[i] = 2

    # Own spot (f5)
    if waiting_is_dumb:
        features[4] = 0
    elif goals[4]:   # own spot is a goal
        features[4] = 2
    
    # 6. Do updates for next state_to_features
    self.previous_collectable_coins = collectable_coins

    return features



def proximity_map (own_position, game_field):
    """
    calculates three values for each tile of the game field:
    1. travel time aka. distance from own position
    2. if tile is reachable from own position or blocked
    3. in which directions one can initially go to reach that tile as quick as possible

    Arguments
    ---------
    own_position : tuple (x, y)
        with x and y being current coordinates coordinates of the agent 
        on the game field. Thus 0 < x < COLS-1, 0 < y < ROWS-1.
    game_field   : np.array, shape = (COLS, ROWS)
        = game_state['field']

    Returns
    -------
    travel_time_map         : np.array, shape like game_field, dtype = int
        Reachable tiles have the value of the number of steps it takes to move to them 
        from own_position.
        Unreachable tiles have the value of DEFAULT_TRAVEL_TIME which is much higher than 
        any reachable time.
    reachable_map           : np.array, shape like game_field, dtype = bool
        A boolean mask of travel_time_map that labels reachable tiles as True and 
        unreachable ones as False.
    original_directions_map : np.array, shape = (COLS, ROWS, 4), dtype = bool
        A map of the game_field that holds a 4-element boolean array for every tile.
        Values of the tile's array correspond to the 4 directions UP, RIGHT, DOWN, LEFT 
        which you might from own_position to reach the tile. Those direction which lead you 
        to reach the tile the fastest are marked True, the others False.
        For example, if you can reach a tile the fastest by either going UP or RIGHT at the step
        then its array will look like this [TRUE, TRUE, FALSE, FALSE].
        This map will be important to quickly find the best direction towards coins, crates,
        opponents and more.
    """


    # Setup of initial values
    distance_map  = np.full_like(game_field, DEFAULT_DISTANCE)
    direction_map = np.full((*game_field.shape, 4), False)

    distance_map[own_position] = 0
    for i, dir in enumerate(DIRECTIONS):
        neighbor = tuple(dir + np.array(own_position))
        if game_field[neighbor] == 0:   # If neighbor is a free field
            direction_map[neighbor][i] = True
    

    # Breadth first search for proximity values to all reachable spots
    frontier = [own_position]
    while len(frontier) > 0:
        current = frontier.pop(0)
        
        for dir in DIRECTIONS:
            neighbor = tuple(dir + np.array(current))
            
            # Update travel time to `neighbor` field
            if game_field[neighbor] == 0:   # If neighbor is a free field
                time = distance_map[current] + 1
                if distance_map[neighbor] > time:
                    distance_map[neighbor] = time
                    frontier.append(neighbor)
                    
                    # Update original direction for `neighbor` field
                    if time > 1:
                        direction_map[neighbor] = direction_map[current]
                        
                # Combine orginial directions if travel times are equal
                elif distance_map[neighbor] == time:
                    direction_map[neighbor] = np.logical_or(
                        direction_map[neighbor], direction_map[current])


    # Derivation of reachability_map
    reachability_map = distance_map != DEFAULT_DISTANCE


    return distance_map, reachability_map, direction_map



def select_reachable (positions, reachability_map):
    """
    """

    if len(positions) > 0:
        positions_array     = np.array(positions)
        positions_tuple     = tuple(positions_array.T)
        reachable_mask      = reachability_map[positions_tuple]
        reachable_positions = positions_array[reachable_mask] 
    else:
        reachable_positions = np.array(positions)

    return reachable_positions



def exclude_from(list, item):
    if item in list:
        smaller_list = list.copy()
        smaller_list.remove(item)
        return smaller_list
    else:
        return list



def select_nearest (positions, distance_map):
    """
    """

    if len(positions) > 0:
        positions_array   = np.array(positions)
        positions_tuple   = tuple(positions_array.T)
        min_distance_mask = distance_map[positions_tuple] == np.amin(distance_map[positions_tuple])
        nearest_positions = positions_array[min_distance_mask]
    else:
        nearest_positions = np.array([])    
    
    return nearest_positions



def make_goals (positions, direction_map, own_position):
    """
    """

    # Direction goals
    goals = np.full(5, False)
    if len(positions) > 0:
        positions_tuple  = tuple(positions.T)
        goal_directions  = direction_map[positions_tuple]
        goals[:4]        = np.any(goal_directions, axis = 0)
        
        # Check if there's a goal on the own_position
        goal_on_own_spot = (np.array(own_position) == positions).all(axis = 1).any()   # numpy-speak for "own_position in position"
        goals[4]         = goal_on_own_spot
    
    return goals



def crate_destruction_map (crate_map, bombs):
    """
    """

    if len(bombs) > 0:
        bomb_array       = np.array([bomb[0] for bomb in bombs])   # Bomb positions
        bomb_tuple       = tuple(bomb_array.T)
        explosion_zones  = np.any(BOMB_MASK[bomb_tuple], axis = 0)   # All fields that will be destroyed due to the current bombs.
        unexploded_zones = np.logical_not(explosion_zones)   # All fields that will be unharmed by the current bombs
    else:
        unexploded_zones = np.full_like(crate_map, True)
    
    crate_mask            = crate_map == 1   # Only show the crate positions
    crates_remaining_mask = np.logical_and(crate_mask, unexploded_zones)
    number_of_crates_destroyed_map \
                          = np.sum(np.logical_and(crates_remaining_mask, BOMB_MASK), axis = (-2, -1))
    
    return number_of_crates_destroyed_map


def sensible_bombing_spots (reachability_map, dumb_bombing_map):
    return reachability_map * np.logical_not(dumb_bombing_map)


def best_crate_bombing_spots (distance_map, number_of_destroyed_crates_map, sensible_bombing_map):
    """
    """

    total_time_map             = distance_map + BOMB_COOLDOWN_TIME   # Time until next bomb can be placed
    reachable_crates_destroyed = sensible_bombing_map * number_of_destroyed_crates_map   # Filtering out the reachable crate_destruction spots (precaution).
    destruction_speed_map      = reachable_crates_destroyed / total_time_map
    max_destruction_speed      = np.amax(destruction_speed_map)
    
    if max_destruction_speed > 0:
        best_spots_mask = np.isclose(destruction_speed_map, max_destruction_speed)   # Safer test for float equality
        best_spots      = np.array(np.where(best_spots_mask)).T
    else:
        best_spots      = np.array([])
    
    return best_spots



def epsilon_greedy (recommended_action, epsilon):
    
    random_action = lambda x: np.random.choice(6) 


    choice = np.random.choice([0, 1], p = [1 - epsilon, epsilon])
    action = [recommended_action, random_action(0)][choice]
    label  = ['policy', 'random'][choice]   # For the debugger
    return (action, label)



def features_to_indices(features):
    """
    Currently works only for feature dimension = 4!
    Currently assigns unique Q-indice to every feature, using order-invariance of action
    Implicitly: A bijective function which maps a value of {0, 1, ..., 15} to each list of order-invariant features

    input: list of 4 features with values in {0, 1, ..., d}, sorted in increasing order
    output: index of feature in a 1D feature matrix 

    """

    index = int(features[0] + features[1]*(features[1]+1)/2 + features[2]*(features[2]+1)*(features[2]+2)/6 
             + features[3]*(features[3]+1)*(features[3]+2)*(features[3]+3)/24)

    # index = features[0]*3**3 + features[1]*3**2 + features[2]*3 + features[3] # was before for 3^4 = 81 different states, not considering symmetry

    return(index)

  

def random_argmax(a):
    """
    Improved np.argmax(a, axis = None):
    Unbiased (i.e. random) selection if mutliple maximal elements.

    Parameters
    ----------
    a : np.array
        1d Array to find indice(s) of maximal value(s).

    Returns
    -------
    np.argmax(a, axis = None) if unique maximal element
    index of one randomly selected maximal element otherwise

    """
    
    max_indices = np.where(a == np.max(a))[0]
    return np.random.choice(max_indices)
