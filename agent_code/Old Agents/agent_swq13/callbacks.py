# Callbacks for agent_swq13
# =========================


import os
import pickle
import random
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


model_name = "swq13_reproduce2"
model_file = f"model_{model_name}.pt"

# Calculating an anealing epsilon
training_rounds        = 100   # Can't this be taken from main?
epsilon_at_last_round  = 0.05   # Set to desired value
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

    features        = state_to_features(game_state)
    sorting_indices = np.argsort(features)   # Moved sorting here to be able to log both sorted and unsorted features.
    sorted_features = features[sorting_indices]
    state_index     = features_to_indices(sorted_features)

    round = game_state['round']
    eps   = epsilon(round)
    if self.train:
        sorted_policy = random_argmax_1d(self.model[state_index])
        policy        = sorting_indices[sorted_policy]  
        action, label = epsilon_greedy(ACTIONS[policy], eps)
        self.state_indices.append(state_index)
        self.sorted_policies.append(sorted_policy)
    else:
        sorted_policy = random_argmax_1d(self.model[state_index])
        policy        = sorting_indices[sorted_policy]
        action        = ACTIONS[policy]
        label         = "policy"

    # Logging
    self.logger.debug(f"act(): Round {round}, Step {game_state['step']}:")
    self.logger.debug(f"act(): Game State: Position {game_state['self'][3]}, Features {features}, epsilon {eps}")
    self.logger.debug(f"act(): Symmetry: Sorted features {sorted_features}, Q-index {state_index}, Sorted policy {sorted_policy}")
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

    ### design features for Task 1 ###
    """
    np.array where each component corresponds to on neighbour field
    = 0 if wall or crate
    = 1 if free
    = 2 if free and (one) nearest field to nearest coin
    """
    X = np.zeros(4, dtype = int) # hand-crafted feature vector
    
    free_space = game_state['field'] == 0 # Boolean numpy array. True for free tiles and False for Crates & Walls
    agent_x, agent_y = game_state['self'][3] # Agent position as coordinates 
    coin_directions = look_for_targets(free_space, (agent_x, agent_y), game_state['coins']) # neighbouring field closest to closest coin
    #print(f"{str(game_state['step']):3}: {str(game_state['self'][3]):8} -> {coin_directions}")

    neighbours = [(agent_x, agent_y - 1), (agent_x + 1, agent_y), 
                  (agent_x, agent_y + 1), (agent_x - 1, agent_y)]   # UP, RIGHT, DOWN, LEFT from (x, y)

    for j, neighbour in enumerate(neighbours):
        if neighbour in coin_directions: 
            X[j] = 2
        elif free_space[neighbour[0], neighbour[1]]:
            X[j] = 1
    
    ### design symmetry transformation for Task 1 ###
    '''
    List of..
    1. unique representative feature of set of unordered features
    2. indices of represented feature
    '''

    return X
    '''
    X_unique =  np.sort(X)
    X_indices = np.argsort(X)

    return([X_unique, X_indices]) 
    '''

def look_for_targets(free_space, start, targets, logger=None):
    """
    Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    
    
    if len(targets) == 0: return []

    frontier    = [start]         # tree leaves
    parent_dict = {start: start}  # branching points
    dist_so_far = {start: 0}      # branch lengths
    best_ones   = []
    next_best   = []   # If no coin is reachable
    best_dist   = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    found_one   = False

    while len(frontier) > 0:   # While there still are reachable tiles
        current = frontier.pop(0)
        
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        current_dist = d + dist_so_far[current]
        if d == 0:   # In case there is a reachable coin, stop only if you have found a path to it.
            # Found path to a target's exact position, mission accomplished!
            best_ones.append(current)
            best_dist = current_dist
            found_one = True
        elif current_dist == best_dist:   # In case no coin is reachable, find reachable tile closest to closest coin.
            next_best.append(current)   
        elif current_dist < best_dist:
            next_best = [current]
            best_dist = current_dist
        
        if found_one and dist_so_far[current] >= best_dist:   # If one target has already been found and this tile doesn't have a target, forget about it.
            # Forget about current tile
            continue    
        else:   # else expand the frontier by adding neighbors        
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y       = current
            directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]   # UP, RIGHT, DOWN, LEFT from (x, y)
            neighbors  = [(x_dir, y_dir)  for (x_dir, y_dir) in directions  if free_space[x_dir, y_dir]]
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
    
    
    if logger: logger.debug(f'Suitable target(s) found at {best_ones}')
    
    # Determine the first step (best direction(s)) towards the best found target tile(s)
    best = best_ones  if found_one  else  next_best
    directions = []
    while len(best) > 0:
        current = best.pop(0)
        parent = parent_dict[current]
        if parent == start:
            if current not in directions:  directions.append(current)
        elif parent not in best:
            best.append(parent)
    return directions
    '''
    for current in best:
        while parent_dict[current] != start:
            current = parent_dict[current]
        if current not in directions:  directions.append(current)
    '''
  


def epsilon_greedy (recommended_action, epsilon):
    
    random_action = lambda x: np.random.choice(ACTIONS[:4])  # don't wait or kill yourself


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
