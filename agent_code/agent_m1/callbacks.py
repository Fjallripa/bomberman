# Callbacks for agent_m1
# =========================


import os
import pickle
import random
import numpy as np

model_name = "m1_10k-no-gamma"
model_file = f"model_{model_name}.pt"


# Global Constants
ACTIONS            = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS         = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])   # UP, RIGHT, DOWN, LEFT
DEFAULT_DISTANCE   = 1000
BOMB_COOLDOWN_TIME = 7
COLS = ROWS        = 17
BLAST              = np.array([-3, -2, -1, 1, 2, 3])

# Calculate constant BOMB_MASK one time
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


# Calculating an anealing epsilon
training_rounds        = 10_000   # Can't this be taken from main?
epsilon_at_last_round  = 0.01   # Set to desired value
epsilon_at_first_round = np.power(epsilon_at_last_round, 1 / training_rounds)  # n-th root of epsilon_at_last_round
epsilon                = lambda round: \
    np.power(epsilon_at_first_round, round)   # does exponentially decrease with training rounds.





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
        # Setup done in setup_training()
        pass

    elif not os.path.isfile(model_file):
        print(f"\nError: the model file {model_file} couldn't be found!\n")

    else:
        self.logger.info(f"Loading model {model_file} from saved state.")
        with open(model_file, "rb") as file:
            self.model = pickle.load(file)



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    Currently: works only for movement actions, i.e. Actions[:4]
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    
    if self.train:  self.timer_act.start()

    features                  = state_to_features(game_state)
    direction_features        = features[:4]
    sorting_indices           = np.argsort(direction_features)   # Moved sorting here to be able to log both sorted and unsorted features.
    sorted_direction_features = direction_features[sorting_indices]
    state_indices             = features_to_indices(sorted_direction_features), features[4], features[5]

    round = game_state['round']
    eps   = epsilon(round)
    if self.train:
        sorted_policy, label = epsilon_greedy(random_argmax_1d(self.model[state_indices]), eps)
        policy = np.concatenate(sorting_indices, np.array([4,5]))[sorted_policy]
        action = ACTIONS[policy]
        self.state_indices.append(state_indices)
        self.sorted_policies.append(sorted_policy)

        '''
        self.unsorted_policies.append(policy) # for debugging purpose
        self.unsorted_features.append(features)
        self.sorted_features.append(sorted_features)
        '''

    else:
        sorted_policy = random_argmax_1d(self.model[state_indices])
        policy        = np.append(sorting_indices, np.array([4,5]))[sorted_policy]
        action        = ACTIONS[policy]
        label         = "policy"

    # Logging
    self.logger.debug(f"act(): Round {round}, Step {game_state['step']}:")
    self.logger.debug(f"act(): Game State: Position {game_state['self'][3]}, Features {features}, epsilon {eps}")
    self.logger.debug(f"act(): Symmetry: Sorted features {np.append(sorted_direction_features, features[4:])}, Q-indeces {state_indices}, Sorted policy {sorted_policy}")
    self.logger.debug(f"act(): Performed {label} action {action}")
    
    # Timing this function
    if self.train: 
        act_time = self.timer_act.stop()
        self.act_times.append(act_time)

    return action 





# Support functions
# -----------------

def state_to_features(game_state: dict) -> np.array:
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


    # 0. relevant game_state info
    own_position      = game_state['self'][3]
    crate_map         = game_state['field']
    collectable_coins = game_state['coins']
    bombs             = game_state['bombs']
    explosion_map     = game_state['explosion_map']

    neighbors = own_position + DIRECTIONS

    
    # 1. Calculate proximity map
    distance_map, reachability_map, direction_map = proximity_map(own_position, crate_map)


    # 2. Check for danger and lethal danger
    going_is_dumb   = np.array([( (not reachability_map[(x,y)]) or explosion_map[(x,y)] ) for [x,y] in neighbors])
    waiting_is_dumb = False
    bombing_is_dumb = False
    if not game_state['self'][2]: 
        bombing_is_dumb = True

    for (bomb_position, bomb_timer) in bombs:
        steps_until_explosion = bomb_timer + 1

        if waiting_is_dumb == False:
            no_future_explosion_mask = np.logical_not(BOMB_MASK[bomb_position])
            rescue_distances = distance_map[reachability_map & no_future_explosion_mask] # improve by including explosions
            minimal_rescue_distance = DEFAULT_DISTANCE if (rescue_distances.size == 0) else np.amin(rescue_distances)
            if steps_until_explosion <= minimal_rescue_distance:
                waiting_is_dumb = True
                bombing_is_dumb = True

        safe_directions = np.amax(direction_map[reachability_map & no_future_explosion_mask & (distance_map <= steps_until_explosion)], axis = 0, initial = False)
        going_is_dumb[np.logical_not(safe_directions)] = True

    if bombing_is_dumb == False:
        no_future_explosion_mask = np.logical_not(BOMB_MASK[own_position])
        rescue_distances = distance_map[reachability_map & no_future_explosion_mask] # improve by including explosions
        minimal_rescue_distance = DEFAULT_DISTANCE if (rescue_distances.size == 0) else np.amin(rescue_distances) 
        if minimal_rescue_distance >= 4:
                bombing_is_dumb = True
    
    
    # 3. Check game mode
    reachable_coins = select_reachable(collectable_coins, reachability_map)
        
    if len(reachable_coins) > 0:
        mode = 0
    else:
        mode = 1

 
    # 4. Compute goal direction
    if mode == 0:
        best_coins      = select_nearest(reachable_coins, distance_map)
        goals           = make_goals(best_coins, direction_map, own_position)

    if mode == 1:
        crates_destroyed = crate_destruction_map(crate_map, bombs)
        best_crate_spots = best_crate_bombing_spots (distance_map, reachability_map, 
                                crates_destroyed, bombing_is_dumb, own_position)
        goals            = make_goals(best_crate_spots, direction_map, own_position)


    # 5. Assemble feature array
    features = np.full(6, 1)
    
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
        
    # Mode (f6)
    features[5] = mode


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



def best_crate_bombing_spots (distance_map, reachability_map, 
        number_of_crates_destroyed_map, bombing_now_is_suicide, own_position):
    """
    """    
    
    if bombing_now_is_suicide:
        reachability_map[own_position] = False   # Exclude own_position from considered bombing spots.

    total_time_map             = distance_map + BOMB_COOLDOWN_TIME   # Time until next bomb can be placed
    reachable_crates_destroyed = reachability_map * number_of_crates_destroyed_map   # Filtering out the reachable crate_destruction spots (precaution).
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

  

def random_argmax_1d(a):
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
