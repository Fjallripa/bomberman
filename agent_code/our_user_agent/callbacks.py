import pickle

game_state_file = lambda x: f'logs/game_states/game_state_{x}.pkl'

def setup(self):
    pass

def act(self, game_state: dict):
    # save game_state
    step = game_state['step']
    with open(game_state_file(step), "wb") as file:
        pickle.dump(game_state, file)

    self.logger.info('Pick action according to pressed key')
    print(repr(game_state['field']))
    return game_state['user_input']
